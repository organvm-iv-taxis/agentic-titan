"""
Firecracker Network Setup

TAP device management and networking configuration for Firecracker VMs.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

logger = logging.getLogger("titan.runtime.firecracker.network")


@dataclass
class NetworkConfig:
    """Network configuration for a VM."""

    tap_device: str = ""
    host_ip: str = "172.16.0.1"
    guest_ip: str = "172.16.0.2"
    netmask: str = "255.255.255.0"
    gateway: str = "172.16.0.1"
    dns: str = "8.8.8.8"
    mtu: int = 1500
    enable_nat: bool = True

    @property
    def cidr(self) -> str:
        """Get CIDR notation for network."""
        # Convert netmask to CIDR
        mask_bits = sum(bin(int(x)).count("1") for x in self.netmask.split("."))
        return f"{self.host_ip}/{mask_bits}"


@dataclass
class TAPDevice:
    """Represents a TAP network device."""

    name: str
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    vm_id: str | None = None
    config: NetworkConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vm_id": self.vm_id,
            "config": self.config.__dict__ if self.config else None,
        }


class FirecrackerNetwork:
    """
    Network management for Firecracker VMs.

    Handles:
    - TAP device creation and cleanup
    - NAT configuration for internet access
    - Network namespace isolation
    - IP address allocation
    """

    def __init__(
        self,
        bridge_name: str = "fcbr0",
        ip_pool_start: str = "172.16.0.0",
        ip_pool_size: int = 256,
    ) -> None:
        self._bridge_name = bridge_name
        self._ip_pool_start = ip_pool_start
        self._ip_pool_size = ip_pool_size
        self._tap_devices: dict[str, TAPDevice] = {}
        self._allocated_ips: set[str] = set()
        self._lock = asyncio.Lock()
        self._next_ip_offset = 2  # Start at .2 (host is .1)

    async def create_tap_device(
        self,
        name: str | None = None,
        vm_id: str | None = None,
    ) -> str:
        """
        Create a TAP device for VM networking.

        Args:
            name: Optional device name (auto-generated if not provided)
            vm_id: Optional VM ID to associate

        Returns:
            TAP device name
        """
        async with self._lock:
            # Generate name if not provided
            if not name:
                name = f"fc-tap-{len(self._tap_devices)}"

            # Check if already exists
            if name in self._tap_devices:
                return name

            try:
                # Create TAP device using ip command
                await self._run_command(f"ip tuntap add dev {name} mode tap")

                # Bring up the device
                await self._run_command(f"ip link set {name} up")

                # Allocate IP for this network
                config = self._allocate_network_config(name)

                # Configure host side IP
                await self._run_command(
                    f"ip addr add {config.host_ip}/{24} dev {name}"
                )

                # Track the device
                tap = TAPDevice(name=name, vm_id=vm_id, config=config)
                self._tap_devices[name] = tap

                logger.info(f"Created TAP device {name} with host IP {config.host_ip}")
                return name

            except Exception as e:
                logger.error(f"Failed to create TAP device {name}: {e}")
                raise

    async def configure_nat(
        self,
        tap_device: str,
        guest_ip: str,
        outbound_iface: str = "eth0",
    ) -> None:
        """
        Configure NAT for VM internet access.

        Args:
            tap_device: TAP device name
            guest_ip: Guest IP address
            outbound_iface: Outbound network interface
        """
        try:
            tap = self._tap_devices.get(tap_device)
            if not tap or not tap.config:
                raise ValueError(f"TAP device {tap_device} not found")

            # Enable IP forwarding
            await self._run_command("sysctl -w net.ipv4.ip_forward=1")

            # Add MASQUERADE rule for outbound traffic
            await self._run_command(
                f"iptables -t nat -A POSTROUTING -o {outbound_iface} "
                f"-s {tap.config.cidr} -j MASQUERADE"
            )

            # Allow forwarding
            await self._run_command(
                f"iptables -A FORWARD -i {tap_device} -o {outbound_iface} -j ACCEPT"
            )
            await self._run_command(
                f"iptables -A FORWARD -i {outbound_iface} -o {tap_device} "
                "-m state --state RELATED,ESTABLISHED -j ACCEPT"
            )

            logger.info(f"Configured NAT for {tap_device}")

        except Exception as e:
            logger.error(f"Failed to configure NAT: {e}")
            raise

    async def cleanup_tap_device(self, name: str) -> None:
        """
        Remove a TAP device.

        Args:
            name: TAP device name
        """
        async with self._lock:
            tap = self._tap_devices.pop(name, None)
            if not tap:
                return

            try:
                # Remove iptables rules if NAT was configured
                if tap.config:
                    try:
                        await self._run_command(
                            f"iptables -t nat -D POSTROUTING "
                            f"-s {tap.config.cidr} -j MASQUERADE",
                            check=False,
                        )
                    except Exception as exc:
                        logger.debug(
                            "Failed to remove NAT rule for %s during cleanup: %s",
                            name,
                            exc,
                        )

                # Delete the device
                await self._run_command(f"ip link delete {name}", check=False)

                # Free allocated IP
                if tap.config:
                    self._allocated_ips.discard(tap.config.guest_ip)

                logger.info(f"Cleaned up TAP device {name}")

            except Exception as e:
                logger.warning(f"Error cleaning up TAP device {name}: {e}")

    @asynccontextmanager
    async def network_namespace(
        self,
        vm_id: str,
    ) -> AsyncGenerator[str, None]:
        """
        Create an isolated network namespace for a VM.

        Args:
            vm_id: VM identifier

        Yields:
            Namespace name
        """
        ns_name = f"fc-ns-{vm_id[:8]}"

        try:
            # Create network namespace
            await self._run_command(f"ip netns add {ns_name}")

            # Create veth pair
            veth_host = f"veth-{vm_id[:6]}-h"
            veth_guest = f"veth-{vm_id[:6]}-g"

            await self._run_command(
                f"ip link add {veth_host} type veth peer name {veth_guest}"
            )

            # Move guest end to namespace
            await self._run_command(f"ip link set {veth_guest} netns {ns_name}")

            # Configure host end
            await self._run_command(f"ip link set {veth_host} up")

            yield ns_name

        finally:
            # Cleanup
            try:
                await self._run_command(f"ip netns delete {ns_name}", check=False)
                await self._run_command(f"ip link delete {veth_host}", check=False)
            except Exception as exc:
                logger.debug(
                    "Failed namespace cleanup for %s: %s",
                    ns_name,
                    exc,
                )

    async def setup_bridge(self) -> None:
        """Create a bridge for VM networking."""
        try:
            # Check if bridge exists
            result = await self._run_command(
                f"ip link show {self._bridge_name}",
                check=False,
            )
            if result.returncode == 0:
                return  # Bridge already exists

            # Create bridge
            await self._run_command(f"ip link add {self._bridge_name} type bridge")
            await self._run_command(f"ip link set {self._bridge_name} up")

            # Assign IP to bridge
            await self._run_command(
                f"ip addr add {self._ip_pool_start.rsplit('.', 1)[0]}.1/24 "
                f"dev {self._bridge_name}"
            )

            logger.info(f"Created bridge {self._bridge_name}")

        except Exception as e:
            logger.error(f"Failed to create bridge: {e}")
            raise

    async def add_to_bridge(self, tap_device: str) -> None:
        """Add a TAP device to the bridge."""
        try:
            await self._run_command(
                f"ip link set {tap_device} master {self._bridge_name}"
            )
            logger.info(f"Added {tap_device} to bridge {self._bridge_name}")
        except Exception as e:
            logger.error(f"Failed to add {tap_device} to bridge: {e}")
            raise

    def get_tap_device(self, name: str) -> TAPDevice | None:
        """Get TAP device info."""
        return self._tap_devices.get(name)

    def list_tap_devices(self) -> list[TAPDevice]:
        """List all TAP devices."""
        return list(self._tap_devices.values())

    def _allocate_network_config(self, tap_name: str) -> NetworkConfig:
        """Allocate network configuration for a TAP device."""
        # Find next available IP
        base = self._ip_pool_start.rsplit(".", 1)[0]

        while self._next_ip_offset < self._ip_pool_size:
            guest_ip = f"{base}.{self._next_ip_offset + 1}"
            host_ip = f"{base}.{self._next_ip_offset}"
            self._next_ip_offset += 2

            if guest_ip not in self._allocated_ips:
                self._allocated_ips.add(guest_ip)
                return NetworkConfig(
                    tap_device=tap_name,
                    host_ip=host_ip,
                    guest_ip=guest_ip,
                )

        raise RuntimeError("IP pool exhausted")

    async def _run_command(
        self,
        cmd: str,
        check: bool = True,
    ) -> asyncio.subprocess.Process:
        """Run a shell command."""
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Command failed: {cmd}\nStderr: {stderr.decode()}"
            )

        # Attach output to process for inspection
        proc.stdout_text = stdout.decode() if stdout else ""  # type: ignore
        proc.stderr_text = stderr.decode() if stderr else ""  # type: ignore

        return proc

    async def cleanup_all(self) -> None:
        """Clean up all network resources."""
        # Copy list to avoid modification during iteration
        devices = list(self._tap_devices.keys())
        for name in devices:
            await self.cleanup_tap_device(name)

        # Remove bridge
        try:
            await self._run_command(
                f"ip link delete {self._bridge_name}",
                check=False,
            )
        except Exception:
            pass

        logger.info("Network cleanup complete")


# Singleton instance
_network_manager: FirecrackerNetwork | None = None


def get_network_manager() -> FirecrackerNetwork:
    """Get the default network manager."""
    global _network_manager
    if _network_manager is None:
        _network_manager = FirecrackerNetwork()
    return _network_manager
