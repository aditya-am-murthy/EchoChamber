"""
Node type definitions for the social network graph
"""

from enum import Enum


class NodeType(Enum):
    """Types of nodes in the social network graph"""
    USER = "user"  # Regular user
    POST = "post"  # Post/status
    NEWS_AGENCY = "news_agency"  # News organization
    INFLUENCER = "influencer"  # High-engagement user
    ORGANIZATION = "organization"  # Other organizations
    MEDIA = "media"  # Media entity
    
    @classmethod
    def from_string(cls, node_type_str: str):
        """Convert string to NodeType"""
        node_type_str = node_type_str.lower().replace(' ', '_')
        for node_type in cls:
            if node_type.value == node_type_str:
                return node_type
        return cls.USER  # Default to USER

