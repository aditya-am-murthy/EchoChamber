"""
Build social network graphs from interaction data using heterogeneous graph structure
for user interaction prediction and virality modeling.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
from .node_types import NodeType


class GraphBuilder:
    """
    Builds heterogeneous graph representations of social network interactions.
    
    Node types: user, post, tag
    Edge types:
        - (user, authors, post): authorship
        - (user, likes, post): like interactions
        - (user, comments, post): comment interactions
        - (post, mentions, user): user mentions in posts
        - (post, has_tag, tag): hashtag associations
        - (post, replies_to, post): reply threads
        - (post, precedes, post): temporal relationships
    """
    
    def __init__(self, include_news_agencies: bool = False):
        """
        Initialize graph builder
        
        Args:
            include_news_agencies: Whether to classify news agency nodes separately
        """
        self.include_news_agencies = include_news_agencies
        
        # Legacy mappings (kept for backward compatibility)
        self.node_id_to_type: Dict[str, NodeType] = {}
        self.node_id_to_index: Dict[str, int] = {}
        self.index_to_node_id: Dict[int, str] = {}
        self.user_interaction_counts: Dict[str, int] = defaultdict(int)
        self.post_engagement: Dict[str, Dict[str, int]] = {}
        
        # New heterogeneous graph mappings
        self.post_ids: List[str] = []
        self.user_ids: Set[str] = set()
        self.tag_ids: Set[str] = set()
        
        self.post_index_map: Dict[str, int] = {}
        self.user_index_map: Dict[str, int] = {}
        self.tag_index_map: Dict[str, int] = {}
        
        # News agency identifiers (can be expanded)
        self.news_agencies = {
            'cnn', 'foxnews', 'fox', 'msnbc', 'abc', 'cbs', 'nbc',
            'reuters', 'ap', 'associated press', 'bbc', 'wsj', 'wall street journal',
            'nytimes', 'new york times', 'washington post', 'wapo',
            'politico', 'the hill', 'breitbart', 'daily wire'
        }
    
    def is_news_agency(self, username: str) -> bool:
        """Check if username belongs to a news agency"""
        username_lower = username.lower()
        return any(agency in username_lower for agency in self.news_agencies)
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load post data from CSV"""
        df = pd.read_csv(csv_path, delimiter=';')
        return df
    
    def _safe_json_list(self, cell) -> List:
        """Safely parse JSON list from cell, handling various formats"""
        if pd.isna(cell) or cell == '':
            return []
        if isinstance(cell, list):
            return cell
        if isinstance(cell, str):
            try:
                parsed = json.loads(cell)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []
    
    def parse_interactions(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Parse likes and comments from a row (legacy method for backward compatibility)
        
        Returns:
            Tuple of (like_usernames, comment_usernames)
        """
        try:
            likes = json.loads(row['likes']) if isinstance(row['likes'], str) else []
            comments = json.loads(row['comments']) if isinstance(row['comments'], str) else []
        except (json.JSONDecodeError, KeyError):
            likes = []
            comments = []
        
        # Extract usernames from comments (comments are tuples of (username, content))
        comment_users = [c[0] if isinstance(c, (list, tuple)) and len(c) > 0 else '' 
                        for c in comments]
        comment_users = [u for u in comment_users if u]
        
        return likes, comment_users
    
    def parse_interactions_v2(self, row: pd.Series) -> Dict:
        """
        Parse all interaction types from new schema with full column support
        
        Returns dict with keys: likes, comments, tags, mentions, media, author, 
        reply_to_id, reply_to_account, created_at, language, visibility
        """
        return {
            'likes': self._safe_json_list(row.get('likes', [])),
            'comments': self._safe_json_list(row.get('comments', [])),
            'tags': self._safe_json_list(row.get('tags', [])),
            'mentions': self._safe_json_list(row.get('mentions', [])),
            'media': self._safe_json_list(row.get('media_attachments', [])),
            'author': str(row.get('author_username', '')) if pd.notna(row.get('author_username')) else '',
            'author_id': str(row.get('author_id', '')) if pd.notna(row.get('author_id')) else '',
            'reply_to_id': row.get('in_reply_to_id') if pd.notna(row.get('in_reply_to_id')) else None,
            'reply_to_account': row.get('in_reply_to_account_id') if pd.notna(row.get('in_reply_to_account_id')) else None,
            'created_at': pd.to_datetime(row.get('created_at'), errors='coerce') if pd.notna(row.get('created_at')) else pd.NaT,
            'language': str(row.get('language', 'unknown')).lower(),
            'visibility': str(row.get('visibility', 'public')).lower()
        }
    
    def build_node_mapping(self, df: pd.DataFrame):
        """Build mapping from node IDs to indices and types"""
        node_counter = 0
        
        # Add post nodes
        for post_id in df['post_id']:
            node_id = f"post_{post_id}"
            if node_id not in self.node_id_to_index:
                self.node_id_to_index[node_id] = node_counter
                self.index_to_node_id[node_counter] = node_id
                self.node_id_to_type[node_id] = NodeType.POST
                node_counter += 1
        
        # Add user nodes from interactions
        for _, row in df.iterrows():
            likes, comment_users = self.parse_interactions(row)
            
            # Add users who liked
            for username in likes:
                if username:
                    node_id = f"user_{username}"
                    if node_id not in self.node_id_to_index:
                        self.node_id_to_index[node_id] = node_counter
                        self.index_to_node_id[node_counter] = node_id
                        
                        # Determine node type
                        if self.include_news_agencies and self.is_news_agency(username):
                            self.node_id_to_type[node_id] = NodeType.NEWS_AGENCY
                        else:
                            self.node_id_to_type[node_id] = NodeType.USER
                        
                        node_counter += 1
                    
                    self.user_interaction_counts[username] += 1
            
            # Add users who commented
            for username in comment_users:
                if username:
                    node_id = f"user_{username}"
                    if node_id not in self.node_id_to_index:
                        self.node_id_to_index[node_id] = node_counter
                        self.index_to_node_id[node_counter] = node_id
                        
                        if self.include_news_agencies and self.is_news_agency(username):
                            self.node_id_to_type[node_id] = NodeType.NEWS_AGENCY
                        else:
                            self.node_id_to_type[node_id] = NodeType.USER
                        
                        node_counter += 1
                    
                    self.user_interaction_counts[username] += 1
        
        # Mark high-engagement users as influencers
        if len(self.user_interaction_counts) > 0:
            threshold = np.percentile(list(self.user_interaction_counts.values()), 90)
            for username, count in self.user_interaction_counts.items():
                node_id = f"user_{username}"
                if node_id in self.node_id_to_type and count >= threshold:
                    if self.node_id_to_type[node_id] == NodeType.USER:
                        self.node_id_to_type[node_id] = NodeType.INFLUENCER
    
    def build_edges(self, df: pd.DataFrame) -> List[Tuple[int, int, Dict]]:
        """
        Build edge list with edge features
        
        Returns:
            List of (source_index, target_index, edge_attrs) tuples
        """
        edges = []
        
        for _, row in df.iterrows():
            post_id = row['post_id']
            post_node_id = f"post_{post_id}"
            post_idx = self.node_id_to_index.get(post_node_id)
            
            if post_idx is None:
                continue
            
            likes, comment_users = self.parse_interactions(row)
            
            # Add like edges (user -> post)
            for username in likes:
                if username:
                    user_node_id = f"user_{username}"
                    user_idx = self.node_id_to_index.get(user_node_id)
                    
                    if user_idx is not None:
                        edges.append((
                            user_idx,
                            post_idx,
                            {'type': 'like', 'weight': 1.0}
                        ))
            
            # Add comment edges (user -> post)
            for username in comment_users:
                if username:
                    user_node_id = f"user_{username}"
                    user_idx = self.node_id_to_index.get(user_node_id)
                    
                    if user_idx is not None:
                        edges.append((
                            user_idx,
                            post_idx,
                            {'type': 'comment', 'weight': 2.0}  # Comments weighted higher
                        ))
        
        return edges
    
    def build_pyg_graph(self, df: pd.DataFrame, node_features: Optional[Dict[str, np.ndarray]] = None) -> Data:
        """
        Build PyTorch Geometric graph
        
        Args:
            df: DataFrame with post data
            node_features: Optional dict mapping node_id to feature vector
        
        Returns:
            PyTorch Geometric Data object
        """
        # Build node mappings
        self.build_node_mapping(df)
        
        # Build edges
        edges = self.build_edges(df)
        
        if not edges:
            # Return empty graph
            num_nodes = len(self.node_id_to_index)
            return Data(
                x=torch.zeros((num_nodes, 768)),  # Default feature size
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=num_nodes
            )
        
        # Extract edge indices and attributes
        edge_index = []
        edge_attr = []
        
        for src, tgt, attrs in edges:
            edge_index.append([src, tgt])
            edge_attr.append([attrs['weight']])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Build node features
        num_nodes = len(self.node_id_to_index)
        if node_features is None:
            # Default features: one-hot encoding of node type
            node_feat_dim = len(NodeType) + 1  # +1 for interaction count
            x = torch.zeros((num_nodes, node_feat_dim))
            
            for node_id, idx in self.node_id_to_index.items():
                node_type = self.node_id_to_type.get(node_id, NodeType.USER)
                # One-hot encode node type
                type_idx = list(NodeType).index(node_type)
                x[idx, type_idx] = 1.0
                
                # Add interaction count (normalized)
                if node_id.startswith('user_'):
                    username = node_id.replace('user_', '')
                    interaction_count = self.user_interaction_counts.get(username, 0)
                    x[idx, -1] = min(interaction_count / 100.0, 1.0)  # Normalize
        else:
            # Use provided features
            feat_dim = next(iter(node_features.values())).shape[0]
            x = torch.zeros((num_nodes, feat_dim))
            
            for node_id, idx in self.node_id_to_index.items():
                if node_id in node_features:
                    x[idx] = torch.tensor(node_features[node_id], dtype=torch.float)
        
        # Build graph
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        # Store metadata
        graph.node_id_to_index = self.node_id_to_index
        graph.index_to_node_id = self.index_to_node_id
        graph.node_id_to_type = self.node_id_to_type
        
        return graph
    
    def get_node_type_features(self) -> Dict[str, np.ndarray]:
        """Get one-hot encoded node type features (legacy method)"""
        features = {}
        num_types = len(NodeType)
        
        for node_id, node_type in self.node_id_to_type.items():
            type_vector = np.zeros(num_types)
            type_idx = list(NodeType).index(node_type)
            type_vector[type_idx] = 1.0
            features[node_id] = type_vector
        
        return features
    
    # ================== NEW HETEROGENEOUS GRAPH METHODS ==================
    
    def collect_nodes(self, df: pd.DataFrame):
        """
        Collect all node IDs by type from dataframe.
        Populates post_ids, user_ids, tag_ids and their index maps.
        """
        self.post_ids = []
        self.user_ids = set()
        self.tag_ids = set()
        
        for _, row in df.iterrows():
            post_id = str(row['post_id'])
            self.post_ids.append(post_id)
            
            interactions = self.parse_interactions_v2(row)
            
            # Collect author
            if interactions['author']:
                self.user_ids.add(interactions['author'])
            
            # Collect likers
            for user in interactions['likes']:
                if user:
                    self.user_ids.add(str(user))
            
            # Collect commenters (handle both list and dict formats)
            for comment in interactions['comments']:
                if isinstance(comment, (list, tuple)) and len(comment) > 0:
                    self.user_ids.add(str(comment[0]))
                elif isinstance(comment, dict) and 'username' in comment:
                    self.user_ids.add(str(comment['username']))
            
            # Collect mentioned users
            for mention in interactions['mentions']:
                if mention:
                    self.user_ids.add(str(mention))
            
            # Collect reply-to account
            if interactions['reply_to_account']:
                self.user_ids.add(str(interactions['reply_to_account']))
            
            # Collect tags (normalize to lowercase)
            for tag in interactions['tags']:
                if tag:
                    self.tag_ids.add(str(tag).lower())
        
        # Build index maps
        self.post_index_map = {pid: i for i, pid in enumerate(self.post_ids)}
        self.user_index_map = {uid: i for i, uid in enumerate(sorted(self.user_ids))}
        self.tag_index_map = {tid: i for i, tid in enumerate(sorted(self.tag_ids))}
    
    def build_node_features_hetero(
        self, 
        df: pd.DataFrame,
        node_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict:
        """
        Build features for all node types in heterogeneous graph.
        
        Args:
            df: DataFrame with post data
            node_embeddings: Optional dict mapping node_id to precomputed embedding
                           (e.g., {'post_12345': array([...]), 'user_john': array([...])})
        
        Returns:
            Dict with keys 'post', 'user', 'tag', each containing 'x' and optionally 'y'
        """
        features = {}
        
        # === POST FEATURES ===
        post_features = []
        post_targets = []  # Virality targets (DO NOT use as features - leakage!)
        
        for post_id in self.post_ids:
            row = df[df['post_id'] == post_id].iloc[0]
            interactions = self.parse_interactions_v2(row)
            
            # Use provided embedding if available
            node_key = f"post_{post_id}"
            if node_embeddings and node_key in node_embeddings:
                post_feat = node_embeddings[node_key]
            else:
                # Fallback: construct basic features from metadata
                # Language one-hot (top 10 languages)
                langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ar']
                lang_vec = np.zeros(len(langs))
                if interactions['language'] in langs:
                    lang_vec[langs.index(interactions['language'])] = 1.0
                
                # Visibility one-hot
                vis_types = ['public', 'unlisted', 'private', 'direct']
                vis_vec = np.zeros(len(vis_types))
                if interactions['visibility'] in vis_types:
                    vis_vec[vis_types.index(interactions['visibility'])] = 1.0
                
                # Media presence
                has_media = 1.0 if len(interactions['media']) > 0 else 0.0
                media_count = min(len(interactions['media']) / 4.0, 1.0)
                
                # Tag/mention counts (normalized)
                tag_count = min(len(interactions['tags']) / 10.0, 1.0)
                mention_count = min(len(interactions['mentions']) / 5.0, 1.0)
                
                # Is reply
                is_reply = 1.0 if interactions['reply_to_id'] is not None else 0.0
                
                # Time features
                if pd.notna(interactions['created_at']):
                    hour = interactions['created_at'].hour / 24.0
                    day = interactions['created_at'].dayofweek / 7.0
                else:
                    hour = day = 0.0
                
                post_feat = np.concatenate([
                    lang_vec,           # 10 dims
                    vis_vec,            # 4 dims
                    [has_media, media_count, tag_count, mention_count, 
                     is_reply, hour, day]  # 7 dims
                ])  # Total: 21 dims
            
            post_features.append(post_feat)
            
            # Store targets separately (for training - NOT features)
            post_targets.append([
                float(row.get('favourites_count', 0)),
                float(row.get('reblogs_count', 0)),
                float(row.get('replies_count', 0))
            ])
        
        features['post'] = {
            'x': np.array(post_features, dtype=np.float32),
            'y': np.array(post_targets, dtype=np.float32)
        }
        
        # === USER FEATURES ===
        # Count interactions per user
        user_interaction_counts = Counter()
        user_like_counts = Counter()
        user_comment_counts = Counter()
        user_author_counts = Counter()
        
        for _, row in df.iterrows():
            interactions = self.parse_interactions_v2(row)
            
            # Count likes
            for user in interactions['likes']:
                if user:
                    user_str = str(user)
                    user_interaction_counts[user_str] += 1
                    user_like_counts[user_str] += 1
            
            # Count comments
            for comment in interactions['comments']:
                if isinstance(comment, (list, tuple)) and len(comment) > 0:
                    user = str(comment[0])
                    user_interaction_counts[user] += 1
                    user_comment_counts[user] += 1
                elif isinstance(comment, dict) and 'username' in comment:
                    user = str(comment['username'])
                    user_interaction_counts[user] += 1
                    user_comment_counts[user] += 1
            
            # Count authored posts
            if interactions['author']:
                user_interaction_counts[interactions['author']] += 1
                user_author_counts[interactions['author']] += 1
        
        # Determine influencers (top 10%)
        counts = list(user_interaction_counts.values())
        influencer_threshold = np.percentile(counts, 90) if counts else 0
        
        user_features = []
        for user_id in sorted(self.user_ids):
            node_key = f"user_{user_id}"
            # Use provided embedding if available
            if node_embeddings and node_key in node_embeddings:
                user_feat = node_embeddings[node_key]
            else:
                total_interactions = user_interaction_counts.get(user_id, 0)
                likes = user_like_counts.get(user_id, 0)
                comments = user_comment_counts.get(user_id, 0)
                authored = user_author_counts.get(user_id, 0)
                
                # Normalized counts
                norm_total = min(total_interactions / 100.0, 1.0)
                like_ratio = likes / max(total_interactions, 1)
                comment_ratio = comments / max(total_interactions, 1)
                author_ratio = authored / max(total_interactions, 1)
                
                # Influencer flag
                is_influencer = 1.0 if total_interactions >= influencer_threshold else 0.0
                
                # News agency flag
                is_news = 1.0 if self.is_news_agency(user_id) else 0.0
                
                user_feat = np.array([
                    norm_total,
                    like_ratio,
                    comment_ratio,
                    author_ratio,
                    is_influencer,
                    is_news
                ], dtype=np.float32)  # 6 dims
            
            user_features.append(user_feat)
        
        features['user'] = {
            'x': np.array(user_features, dtype=np.float32)
        }
        
        # === TAG FEATURES ===
        tag_counts = Counter()
        for _, row in df.iterrows():
            for tag in self._safe_json_list(row.get('tags', [])):
                if tag:
                    tag_counts[str(tag).lower()] += 1
        
        tag_features = []
        for tag_id in sorted(self.tag_ids):
            node_key = f"tag_{tag_id}"
            if node_embeddings and node_key in node_embeddings:
                tag_feat = node_embeddings[node_key]
            else:
                count = tag_counts.get(tag_id, 0)
                norm_count = min(count / 50.0, 1.0)
                tag_feat = np.array([norm_count], dtype=np.float32)
            tag_features.append(tag_feat)
        
        features['tag'] = {
            'x': np.array(tag_features, dtype=np.float32)
        }
        
        return features
    
    def build_temporal_edges(
        self,
        df: pd.DataFrame,
        window: int = 5,
        tau_hours: float = 12.0,
        max_hours: float = 72.0
    ) -> List[Tuple[int, int, float]]:
        """
        Build temporal edges between posts based on created_at timestamps.
        
        Args:
            df: DataFrame with post data
            window: Number of previous posts to connect
            tau_hours: Decay time constant (smaller = faster decay)
            max_hours: Maximum time gap to consider
        
        Returns:
            List of (src_post_idx, tgt_post_idx, weight) tuples
        """
        # Sort posts by time
        df_sorted = df.copy()
        df_sorted['created_at_parsed'] = pd.to_datetime(
            df_sorted['created_at'], errors='coerce'
        )
        df_sorted = df_sorted.sort_values('created_at_parsed')
        
        edges = []
        posts = []
        
        for _, row in df_sorted.iterrows():
            post_id = str(row['post_id'])
            post_idx = self.post_index_map.get(post_id)
            created_at = row['created_at_parsed']
            
            if post_idx is not None:
                posts.append((post_idx, created_at))
        
        # Sliding window: connect each post to previous `window` posts
        for i in range(len(posts)):
            tgt_idx, tgt_time = posts[i]
            
            for j in range(max(0, i - window), i):
                src_idx, src_time = posts[j]
                
                # Calculate time delta
                if pd.notna(src_time) and pd.notna(tgt_time):
                    delta_hours = (tgt_time - src_time).total_seconds() / 3600.0
                else:
                    delta_hours = float(i - j)  # Fallback to index distance
                
                if delta_hours > max_hours:
                    continue
                
                # Exponential decay weight
                weight = float(np.exp(-delta_hours / tau_hours))
                
                if weight > 0.01:  # Threshold to avoid tiny weights
                    edges.append((src_idx, tgt_idx, weight))
        
        return edges
    
    def build_hetero_edges(self, df: pd.DataFrame) -> Dict:
        """
        Build all edge types for heterogeneous graph.
        
        Returns dict with edge type tuples as keys, containing 'index' and 'attr' lists.
        """
        edges = {
            ('user', 'authors', 'post'): {'index': [], 'attr': []},
            ('user', 'likes', 'post'): {'index': [], 'attr': []},
            ('user', 'comments', 'post'): {'index': [], 'attr': []},
            ('post', 'mentions', 'user'): {'index': [], 'attr': []},
            ('post', 'has_tag', 'tag'): {'index': [], 'attr': []},
            ('post', 'replies_to', 'post'): {'index': [], 'attr': []},
            ('post', 'precedes', 'post'): {'index': [], 'attr': []},
        }
        
        # Process each post
        for _, row in df.iterrows():
            post_id = str(row['post_id'])
            post_idx = self.post_index_map.get(post_id)
            if post_idx is None:
                continue
            
            interactions = self.parse_interactions_v2(row)
            
            # 1. AUTHOR EDGE (user -> post)
            if interactions['author']:
                author_idx = self.user_index_map.get(interactions['author'])
                if author_idx is not None:
                    edges[('user', 'authors', 'post')]['index'].append([author_idx, post_idx])
                    edges[('user', 'authors', 'post')]['attr'].append([1.0])
            
            # 2. LIKE EDGES (user -> post)
            for user in interactions['likes']:
                if user:
                    user_idx = self.user_index_map.get(str(user))
                    if user_idx is not None:
                        edges[('user', 'likes', 'post')]['index'].append([user_idx, post_idx])
                        edges[('user', 'likes', 'post')]['attr'].append([1.0])
            
            # 3. COMMENT EDGES (user -> post)
            for comment in interactions['comments']:
                if isinstance(comment, (list, tuple)) and len(comment) > 0:
                    user = str(comment[0])
                elif isinstance(comment, dict) and 'username' in comment:
                    user = str(comment['username'])
                else:
                    continue
                
                user_idx = self.user_index_map.get(user)
                if user_idx is not None:
                    edges[('user', 'comments', 'post')]['index'].append([user_idx, post_idx])
                    edges[('user', 'comments', 'post')]['attr'].append([2.0])  # Higher weight
            
            # 4. MENTION EDGES (post -> user)
            for mention in interactions['mentions']:
                if mention:
                    user_idx = self.user_index_map.get(str(mention))
                    if user_idx is not None:
                        edges[('post', 'mentions', 'user')]['index'].append([post_idx, user_idx])
                        edges[('post', 'mentions', 'user')]['attr'].append([1.0])
            
            # 5. TAG EDGES (post -> tag)
            for tag in interactions['tags']:
                if tag:
                    tag_idx = self.tag_index_map.get(str(tag).lower())
                    if tag_idx is not None:
                        edges[('post', 'has_tag', 'tag')]['index'].append([post_idx, tag_idx])
                        edges[('post', 'has_tag', 'tag')]['attr'].append([1.0])
            
            # 6. REPLY EDGES (original_post -> reply_post)
            if interactions['reply_to_id'] is not None:
                reply_to_idx = self.post_index_map.get(str(interactions['reply_to_id']))
                if reply_to_idx is not None:
                    edges[('post', 'replies_to', 'post')]['index'].append([reply_to_idx, post_idx])
                    edges[('post', 'replies_to', 'post')]['attr'].append([1.0])
        
        # 7. TEMPORAL EDGES (post -> post)
        temporal_edges = self.build_temporal_edges(df, window=5, tau_hours=12.0)
        for src_idx, tgt_idx, weight in temporal_edges:
            edges[('post', 'precedes', 'post')]['index'].append([src_idx, tgt_idx])
            edges[('post', 'precedes', 'post')]['attr'].append([weight])
        
        return edges
    
    def build_hetero_graph(
        self,
        df: pd.DataFrame,
        node_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> HeteroData:
        """
        Build complete heterogeneous graph for user interaction prediction and virality modeling.
        
        Args:
            df: DataFrame with columns: post_id, content, created_at, author_username, 
                author_id, favourites_count, reblogs_count, replies_count, language, 
                visibility, in_reply_to_id, in_reply_to_account_id, url, tags, 
                mentions, media_attachments, pinned, likes, comments
            node_embeddings: Optional precomputed embeddings dict
        
        Returns:
            HeteroData object with node types (user, post, tag) and edge types
            (authors, likes, comments, mentions, has_tag, replies_to, precedes)
        """
        # Collect nodes
        self.collect_nodes(df)
        
        # Build features
        features = self.build_node_features_hetero(df, node_embeddings)
        
        # Build edges
        edges = self.build_hetero_edges(df)
        
        # Assemble HeteroData
        hetero = HeteroData()
        
        # Add node features
        hetero['post'].x = torch.tensor(features['post']['x'], dtype=torch.float)
        hetero['post'].y = torch.tensor(features['post']['y'], dtype=torch.float)  # Targets
        hetero['user'].x = torch.tensor(features['user']['x'], dtype=torch.float)
        hetero['tag'].x = torch.tensor(features['tag']['x'], dtype=torch.float)
        
        # Add edges
        for edge_type, edge_data in edges.items():
            if edge_data['index']:
                edge_index = torch.tensor(edge_data['index'], dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_data['attr'], dtype=torch.float)
                
                hetero[edge_type].edge_index = edge_index
                hetero[edge_type].edge_attr = edge_attr
        
        # Store metadata
        hetero.post_index_map = self.post_index_map
        hetero.user_index_map = self.user_index_map
        hetero.tag_index_map = self.tag_index_map
        hetero.post_ids = self.post_ids
        hetero.user_ids = sorted(self.user_ids)
        hetero.tag_ids = sorted(self.tag_ids)
        
        return hetero

