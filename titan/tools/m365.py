"""
Microsoft 365 Tool - Graph API integration for M365 services.

Provides:
- Email (Outlook) operations
- Calendar management
- OneDrive file access
- Teams messaging
- SharePoint document access

Reference: vendor/agents/m365-agent/
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.tools.m365")


# ============================================================================
# Data Structures
# ============================================================================


class M365Service(str, Enum):
    """Microsoft 365 services."""
    OUTLOOK = "outlook"
    CALENDAR = "calendar"
    ONEDRIVE = "onedrive"
    TEAMS = "teams"
    SHAREPOINT = "sharepoint"


@dataclass
class M365User:
    """Microsoft 365 user."""
    id: str
    display_name: str
    email: str
    job_title: str | None = None
    department: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "email": self.email,
            "job_title": self.job_title,
            "department": self.department,
        }


@dataclass
class EmailMessage:
    """Email message."""
    id: str
    subject: str
    body: str
    body_preview: str
    sender: M365User | None = None
    recipients: list[M365User] = field(default_factory=list)
    cc: list[M365User] = field(default_factory=list)
    received_datetime: datetime | None = None
    is_read: bool = False
    has_attachments: bool = False
    importance: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "body_preview": self.body_preview,
            "sender": self.sender.to_dict() if self.sender else None,
            "recipients": [r.to_dict() for r in self.recipients],
            "received_datetime": self.received_datetime.isoformat() if self.received_datetime else None,
            "is_read": self.is_read,
            "has_attachments": self.has_attachments,
            "importance": self.importance,
        }


@dataclass
class CalendarEvent:
    """Calendar event."""
    id: str
    subject: str
    body: str = ""
    start: datetime | None = None
    end: datetime | None = None
    location: str = ""
    is_all_day: bool = False
    attendees: list[M365User] = field(default_factory=list)
    organizer: M365User | None = None
    is_online_meeting: bool = False
    online_meeting_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "body": self.body,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "location": self.location,
            "is_all_day": self.is_all_day,
            "attendees": [a.to_dict() for a in self.attendees],
            "organizer": self.organizer.to_dict() if self.organizer else None,
            "is_online_meeting": self.is_online_meeting,
            "online_meeting_url": self.online_meeting_url,
        }


@dataclass
class DriveItem:
    """OneDrive/SharePoint item."""
    id: str
    name: str
    path: str
    is_folder: bool = False
    size: int = 0
    created_datetime: datetime | None = None
    modified_datetime: datetime | None = None
    web_url: str | None = None
    download_url: str | None = None
    mime_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "is_folder": self.is_folder,
            "size": self.size,
            "created_datetime": self.created_datetime.isoformat() if self.created_datetime else None,
            "modified_datetime": self.modified_datetime.isoformat() if self.modified_datetime else None,
            "web_url": self.web_url,
            "mime_type": self.mime_type,
        }


@dataclass
class TeamsMessage:
    """Teams chat message."""
    id: str
    content: str
    sender: M365User | None = None
    created_datetime: datetime | None = None
    chat_id: str | None = None
    channel_id: str | None = None
    team_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "sender": self.sender.to_dict() if self.sender else None,
            "created_datetime": self.created_datetime.isoformat() if self.created_datetime else None,
            "chat_id": self.chat_id,
            "channel_id": self.channel_id,
            "team_id": self.team_id,
        }


# ============================================================================
# Microsoft Graph Client
# ============================================================================


class GraphClient:
    """
    Microsoft Graph API client.

    Handles authentication and API requests to Microsoft 365 services.
    """

    GRAPH_URL = "https://graph.microsoft.com/v1.0"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        tenant_id: str | None = None,
        access_token: str | None = None,
    ) -> None:
        """
        Initialize Graph client.

        Args:
            client_id: Azure AD app client ID
            client_secret: Azure AD app client secret
            tenant_id: Azure AD tenant ID
            access_token: Pre-obtained access token
        """
        self.client_id = client_id or os.environ.get("AZURE_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("AZURE_CLIENT_SECRET")
        self.tenant_id = tenant_id or os.environ.get("AZURE_TENANT_ID")
        self._access_token = access_token
        self._token_expires: datetime | None = None

    async def _get_token(self) -> str:
        """Get access token, refreshing if needed."""
        if self._access_token and self._token_expires:
            if datetime.now() < self._token_expires - timedelta(minutes=5):
                return self._access_token

        # Get new token
        if not all([self.client_id, self.client_secret, self.tenant_id]):
            raise ValueError("Azure AD credentials not configured")

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": "https://graph.microsoft.com/.default",
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"Token request failed: {response.text}")

            data = response.json()
            self._access_token = data["access_token"]
            self._token_expires = datetime.now() + timedelta(seconds=data["expires_in"])

        return self._access_token

    async def request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a Graph API request."""
        import httpx

        token = await self._get_token()  # allow-secret

        url = f"{self.GRAPH_URL}{endpoint}"

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                params=params,
                json=json,
                timeout=30.0,
            )

            if response.status_code >= 400:
                logger.error(f"Graph API error: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}

            if response.status_code == 204:
                return {"success": True}

            return response.json()

    async def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET request."""
        return await self.request("GET", endpoint, params=params)

    async def post(self, endpoint: str, json: dict[str, Any]) -> dict[str, Any]:
        """POST request."""
        return await self.request("POST", endpoint, json=json)

    async def patch(self, endpoint: str, json: dict[str, Any]) -> dict[str, Any]:
        """PATCH request."""
        return await self.request("PATCH", endpoint, json=json)

    async def delete(self, endpoint: str) -> dict[str, Any]:
        """DELETE request."""
        return await self.request("DELETE", endpoint)


# ============================================================================
# Service Handlers
# ============================================================================


class OutlookService:
    """Outlook mail service."""

    def __init__(self, client: GraphClient) -> None:
        self.client = client

    async def list_messages(
        self,
        user_id: str = "me",
        folder: str = "inbox",
        top: int = 10,
        filter_query: str | None = None,
    ) -> list[EmailMessage]:
        """List email messages."""
        endpoint = f"/users/{user_id}/mailFolders/{folder}/messages"
        params = {
            "$top": top,
            "$orderby": "receivedDateTime desc",
            "$select": "id,subject,bodyPreview,from,toRecipients,ccRecipients,receivedDateTime,isRead,hasAttachments,importance",
        }
        if filter_query:
            params["$filter"] = filter_query

        response = await self.client.get(endpoint, params)

        if "error" in response:
            logger.error(f"Failed to list messages: {response['error']}")
            return []

        messages = []
        for item in response.get("value", []):
            sender = None
            if item.get("from"):
                sender = M365User(
                    id=item["from"]["emailAddress"].get("address", ""),
                    display_name=item["from"]["emailAddress"].get("name", ""),
                    email=item["from"]["emailAddress"].get("address", ""),
                )

            messages.append(EmailMessage(
                id=item["id"],
                subject=item.get("subject", ""),
                body="",
                body_preview=item.get("bodyPreview", ""),
                sender=sender,
                received_datetime=datetime.fromisoformat(item["receivedDateTime"].replace("Z", "+00:00")) if item.get("receivedDateTime") else None,
                is_read=item.get("isRead", False),
                has_attachments=item.get("hasAttachments", False),
                importance=item.get("importance", "normal"),
            ))

        return messages

    async def get_message(self, message_id: str, user_id: str = "me") -> EmailMessage | None:
        """Get a specific message."""
        response = await self.client.get(f"/users/{user_id}/messages/{message_id}")

        if "error" in response:
            return None

        sender = None
        if response.get("from"):
            sender = M365User(
                id=response["from"]["emailAddress"].get("address", ""),
                display_name=response["from"]["emailAddress"].get("name", ""),
                email=response["from"]["emailAddress"].get("address", ""),
            )

        return EmailMessage(
            id=response["id"],
            subject=response.get("subject", ""),
            body=response.get("body", {}).get("content", ""),
            body_preview=response.get("bodyPreview", ""),
            sender=sender,
            received_datetime=datetime.fromisoformat(response["receivedDateTime"].replace("Z", "+00:00")) if response.get("receivedDateTime") else None,
            is_read=response.get("isRead", False),
            has_attachments=response.get("hasAttachments", False),
        )

    async def send_message(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        user_id: str = "me",
    ) -> bool:
        """Send an email message."""
        message = {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": body,
            },
            "toRecipients": [
                {"emailAddress": {"address": email}}
                for email in to
            ],
        }

        if cc:
            message["ccRecipients"] = [
                {"emailAddress": {"address": email}}
                for email in cc
            ]

        response = await self.client.post(
            f"/users/{user_id}/sendMail",
            json={"message": message},
        )

        return "error" not in response


class CalendarService:
    """Calendar service."""

    def __init__(self, client: GraphClient) -> None:
        self.client = client

    async def list_events(
        self,
        user_id: str = "me",
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        top: int = 10,
    ) -> list[CalendarEvent]:
        """List calendar events."""
        params: dict[str, Any] = {
            "$top": top,
            "$orderby": "start/dateTime",
        }

        if start_datetime and end_datetime:
            params["$filter"] = (
                f"start/dateTime ge '{start_datetime.isoformat()}' and "
                f"end/dateTime le '{end_datetime.isoformat()}'"
            )

        response = await self.client.get(f"/users/{user_id}/events", params)

        if "error" in response:
            return []

        events = []
        for item in response.get("value", []):
            events.append(CalendarEvent(
                id=item["id"],
                subject=item.get("subject", ""),
                body=item.get("body", {}).get("content", ""),
                start=datetime.fromisoformat(item["start"]["dateTime"]) if item.get("start") else None,
                end=datetime.fromisoformat(item["end"]["dateTime"]) if item.get("end") else None,
                location=item.get("location", {}).get("displayName", ""),
                is_all_day=item.get("isAllDay", False),
                is_online_meeting=item.get("isOnlineMeeting", False),
                online_meeting_url=item.get("onlineMeeting", {}).get("joinUrl"),
            ))

        return events

    async def create_event(
        self,
        subject: str,
        start: datetime,
        end: datetime,
        attendees: list[str] | None = None,
        body: str = "",
        location: str = "",
        is_online_meeting: bool = False,
        user_id: str = "me",
    ) -> CalendarEvent | None:
        """Create a calendar event."""
        event_data: dict[str, Any] = {
            "subject": subject,
            "start": {
                "dateTime": start.isoformat(),
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": end.isoformat(),
                "timeZone": "UTC",
            },
        }

        if body:
            event_data["body"] = {"contentType": "HTML", "content": body}
        if location:
            event_data["location"] = {"displayName": location}
        if attendees:
            event_data["attendees"] = [
                {"emailAddress": {"address": email}, "type": "required"}
                for email in attendees
            ]
        if is_online_meeting:
            event_data["isOnlineMeeting"] = True
            event_data["onlineMeetingProvider"] = "teamsForBusiness"

        response = await self.client.post(f"/users/{user_id}/events", event_data)

        if "error" in response:
            return None

        return CalendarEvent(
            id=response["id"],
            subject=response.get("subject", ""),
            start=datetime.fromisoformat(response["start"]["dateTime"]) if response.get("start") else None,
            end=datetime.fromisoformat(response["end"]["dateTime"]) if response.get("end") else None,
            is_online_meeting=response.get("isOnlineMeeting", False),
            online_meeting_url=response.get("onlineMeeting", {}).get("joinUrl"),
        )


class OneDriveService:
    """OneDrive service."""

    def __init__(self, client: GraphClient) -> None:
        self.client = client

    async def list_items(
        self,
        path: str = "/",
        user_id: str = "me",
        top: int = 100,
    ) -> list[DriveItem]:
        """List items in a folder."""
        if path == "/":
            endpoint = f"/users/{user_id}/drive/root/children"
        else:
            endpoint = f"/users/{user_id}/drive/root:/{path}:/children"

        response = await self.client.get(endpoint, {"$top": top})

        if "error" in response:
            return []

        items = []
        for item in response.get("value", []):
            items.append(DriveItem(
                id=item["id"],
                name=item["name"],
                path=item.get("parentReference", {}).get("path", "") + "/" + item["name"],
                is_folder="folder" in item,
                size=item.get("size", 0),
                created_datetime=datetime.fromisoformat(item["createdDateTime"].replace("Z", "+00:00")) if item.get("createdDateTime") else None,
                modified_datetime=datetime.fromisoformat(item["lastModifiedDateTime"].replace("Z", "+00:00")) if item.get("lastModifiedDateTime") else None,
                web_url=item.get("webUrl"),
                mime_type=item.get("file", {}).get("mimeType"),
            ))

        return items

    async def get_item(
        self,
        path: str,
        user_id: str = "me",
    ) -> DriveItem | None:
        """Get item metadata."""
        response = await self.client.get(f"/users/{user_id}/drive/root:/{path}")

        if "error" in response:
            return None

        return DriveItem(
            id=response["id"],
            name=response["name"],
            path=path,
            is_folder="folder" in response,
            size=response.get("size", 0),
            web_url=response.get("webUrl"),
            download_url=response.get("@microsoft.graph.downloadUrl"),
        )

    async def download_item(
        self,
        path: str,
        user_id: str = "me",
    ) -> bytes | None:
        """Download file content."""
        import httpx

        item = await self.get_item(path, user_id)
        if not item or not item.download_url:
            return None

        async with httpx.AsyncClient() as client:
            response = await client.get(item.download_url)
            if response.status_code == 200:
                return response.content

        return None

    async def upload_item(
        self,
        path: str,
        content: bytes,
        user_id: str = "me",
    ) -> DriveItem | None:
        """Upload a file."""
        import httpx

        token = await self.client._get_token()  # allow-secret

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.client.GRAPH_URL}/users/{user_id}/drive/root:/{path}:/content",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                },
                content=content,
            )

            if response.status_code not in (200, 201):
                return None

            data = response.json()
            return DriveItem(
                id=data["id"],
                name=data["name"],
                path=path,
                size=data.get("size", 0),
                web_url=data.get("webUrl"),
            )


class TeamsService:
    """Microsoft Teams service."""

    def __init__(self, client: GraphClient) -> None:
        self.client = client

    async def list_teams(self, user_id: str = "me") -> list[dict[str, Any]]:
        """List user's teams."""
        response = await self.client.get(f"/users/{user_id}/joinedTeams")

        if "error" in response:
            return []

        return [
            {
                "id": team["id"],
                "display_name": team.get("displayName", ""),
                "description": team.get("description", ""),
            }
            for team in response.get("value", [])
        ]

    async def list_channels(self, team_id: str) -> list[dict[str, Any]]:
        """List channels in a team."""
        response = await self.client.get(f"/teams/{team_id}/channels")

        if "error" in response:
            return []

        return [
            {
                "id": channel["id"],
                "display_name": channel.get("displayName", ""),
                "description": channel.get("description", ""),
            }
            for channel in response.get("value", [])
        ]

    async def send_channel_message(
        self,
        team_id: str,
        channel_id: str,
        content: str,
    ) -> TeamsMessage | None:
        """Send a message to a channel."""
        response = await self.client.post(
            f"/teams/{team_id}/channels/{channel_id}/messages",
            json={
                "body": {
                    "content": content,
                }
            },
        )

        if "error" in response:
            return None

        return TeamsMessage(
            id=response["id"],
            content=content,
            team_id=team_id,
            channel_id=channel_id,
        )


# ============================================================================
# Microsoft 365 Tool
# ============================================================================


class Microsoft365Tool:
    """
    Tool for Microsoft 365 integration.

    Provides access to Outlook, Calendar, OneDrive, and Teams.

    Example:
        tool = Microsoft365Tool()

        # List emails
        emails = await tool.execute("list_emails", folder="inbox", top=5)

        # Create calendar event
        event = await tool.execute(
            "create_event",
            subject="Team Meeting",
            start="2024-01-15T10:00:00",
            end="2024-01-15T11:00:00",
            is_online_meeting=True,
        )
    """

    name = "microsoft_365"
    description = "Access Microsoft 365 services: Outlook, Calendar, OneDrive, Teams"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        tenant_id: str | None = None,
        access_token: str | None = None,
    ) -> None:
        self.client = GraphClient(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            access_token=access_token,
        )

        # Initialize services
        self.outlook = OutlookService(self.client)
        self.calendar = CalendarService(self.client)
        self.onedrive = OneDriveService(self.client)
        self.teams = TeamsService(self.client)

    async def execute(
        self,
        action: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute a tool action.

        Outlook Actions:
            - list_emails: List emails from a folder
            - get_email: Get a specific email
            - send_email: Send an email

        Calendar Actions:
            - list_events: List calendar events
            - create_event: Create a calendar event

        OneDrive Actions:
            - list_files: List files in a folder
            - get_file: Get file metadata
            - download_file: Download file content
            - upload_file: Upload a file

        Teams Actions:
            - list_teams: List user's teams
            - list_channels: List channels in a team
            - send_message: Send a message to a channel
        """
        try:
            # Outlook
            if action == "list_emails":
                messages = await self.outlook.list_messages(
                    folder=kwargs.get("folder", "inbox"),
                    top=kwargs.get("top", 10),
                    filter_query=kwargs.get("filter"),
                )
                return {"emails": [m.to_dict() for m in messages]}

            elif action == "get_email":
                message = await self.outlook.get_message(kwargs["message_id"])
                return {"email": message.to_dict() if message else None}

            elif action == "send_email":
                success = await self.outlook.send_message(
                    to=kwargs["to"],
                    subject=kwargs["subject"],
                    body=kwargs["body"],
                    cc=kwargs.get("cc"),
                )
                return {"success": success}

            # Calendar
            elif action == "list_events":
                start = datetime.fromisoformat(kwargs["start"]) if kwargs.get("start") else None
                end = datetime.fromisoformat(kwargs["end"]) if kwargs.get("end") else None
                events = await self.calendar.list_events(
                    start_datetime=start,
                    end_datetime=end,
                    top=kwargs.get("top", 10),
                )
                return {"events": [e.to_dict() for e in events]}

            elif action == "create_event":
                event = await self.calendar.create_event(
                    subject=kwargs["subject"],
                    start=datetime.fromisoformat(kwargs["start"]),
                    end=datetime.fromisoformat(kwargs["end"]),
                    attendees=kwargs.get("attendees"),
                    body=kwargs.get("body", ""),
                    location=kwargs.get("location", ""),
                    is_online_meeting=kwargs.get("is_online_meeting", False),
                )
                return {"event": event.to_dict() if event else None}

            # OneDrive
            elif action == "list_files":
                items = await self.onedrive.list_items(
                    path=kwargs.get("path", "/"),
                    top=kwargs.get("top", 100),
                )
                return {"files": [i.to_dict() for i in items]}

            elif action == "get_file":
                item = await self.onedrive.get_item(kwargs["path"])
                return {"file": item.to_dict() if item else None}

            elif action == "download_file":
                content = await self.onedrive.download_item(kwargs["path"])
                if content:
                    import base64
                    return {
                        "content": base64.b64encode(content).decode("utf-8"),
                        "size": len(content),
                    }
                return {"error": "File not found"}

            elif action == "upload_file":
                import base64
                content = base64.b64decode(kwargs["content"])
                item = await self.onedrive.upload_item(kwargs["path"], content)
                return {"file": item.to_dict() if item else None}

            # Teams
            elif action == "list_teams":
                teams = await self.teams.list_teams()
                return {"teams": teams}

            elif action == "list_channels":
                channels = await self.teams.list_channels(kwargs["team_id"])
                return {"channels": channels}

            elif action == "send_message":
                message = await self.teams.send_channel_message(
                    team_id=kwargs["team_id"],
                    channel_id=kwargs["channel_id"],
                    content=kwargs["content"],
                )
                return {"message": message.to_dict() if message else None}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"M365 action failed: {e}")
            return {"error": str(e)}


# ============================================================================
# Convenience Functions
# ============================================================================


_m365_tool: Microsoft365Tool | None = None


def get_m365_tool() -> Microsoft365Tool:
    """Get the global M365 tool instance."""
    global _m365_tool
    if _m365_tool is None:
        _m365_tool = Microsoft365Tool()
    return _m365_tool
