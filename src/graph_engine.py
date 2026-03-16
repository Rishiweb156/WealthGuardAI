"""Graph RAG Engine - Relationship Discovery in Financial Data.

Uses NetworkX to build knowledge graphs of spending patterns,
identifying hidden relationships between merchants, categories, and time.
"""
import logging
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GraphNode(BaseModel):
    """Node in the financial graph."""
    id: str
    type: str  # merchant, category, time_period
    weight: float


class GraphEdge(BaseModel):
    """Edge connecting nodes."""
    source: str
    target: str
    weight: float
    transaction_count: int


class GraphInsight(BaseModel):
    """Insights from graph analysis."""
    top_merchants: List[Tuple[str, float]]
    spending_clusters: List[Dict]
    influential_categories: List[Tuple[str, float]]
    anomalous_connections: List[Dict]
    summary: str


class FinancialGraph:
    """Graph-based financial analysis engine."""
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.merchant_category_graph = nx.Graph()
        
    def build_graph(self, df: pd.DataFrame) -> None:
        """Build comprehensive knowledge graph from transactions.
        
        Creates a multi-layer graph:
        - Merchants → Categories (spending distribution)
        - Categories → Time Periods (temporal patterns)
        - Merchants → Time Periods (merchant loyalty)
        """
        logger.info("Building financial knowledge graph...")
        
        # Ensure required columns exist
        required_cols = ["Narration", "category", "Withdrawal (INR)", "parsed_date"]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                return
        
        # Extract time features
        df['parsed_date'] = pd.to_datetime(df['parsed_date'], errors='coerce')
        df['month'] = df['parsed_date'].dt.to_period('M').astype(str)
        df['day_of_week'] = df['parsed_date'].dt.day_name()
        df['hour'] = df['parsed_date'].dt.hour
        
        # Build merchant-category graph
        for _, row in df.iterrows():
            merchant = str(row.get("Narration", "Unknown"))[:50]  # Truncate long names
            category = str(row.get("category", "Uncategorized"))
            amount = float(row.get("Withdrawal (INR)", 0))
            month = str(row.get("month", "Unknown"))
            day = str(row.get("day_of_week", "Unknown"))
            
            if amount <= 0:
                continue
            
            # Add nodes
            self.G.add_node(merchant, type="merchant", total_spent=0)
            self.G.add_node(category, type="category", total_received=0)
            self.G.add_node(f"month_{month}", type="time", total_flow=0)
            self.G.add_node(f"day_{day}", type="day", total_flow=0)
            
            # Update node attributes
            self.G.nodes[merchant]['total_spent'] = self.G.nodes[merchant].get('total_spent', 0) + amount
            self.G.nodes[category]['total_received'] = self.G.nodes[category].get('total_received', 0) + amount
            
            # Add edges with weights
            self._add_or_update_edge(merchant, category, amount)
            self._add_or_update_edge(category, f"month_{month}", amount)
            self._add_or_update_edge(merchant, f"day_{day}", amount)
        
        logger.info(f"Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def _add_or_update_edge(self, source: str, target: str, amount: float) -> None:
        """Add or update edge between nodes."""
        if self.G.has_edge(source, target):
            self.G[source][target]['weight'] += amount
            self.G[source][target]['count'] += 1
        else:
            self.G.add_edge(source, target, weight=amount, count=1)
    
    def detect_top_merchants(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find merchants with highest spending impact using PageRank.
        
        PageRank identifies influential nodes in the graph - merchants that
        not only receive high spending but are connected to high-spending categories.
        """
        try:
            # Filter merchant nodes
            merchant_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'merchant']
            
            if not merchant_nodes:
                return []
            
            # Calculate PageRank (influence)
            pagerank = nx.pagerank(self.G, weight='weight')
            
            # Filter and sort merchants
            merchant_ranks = [(node, pagerank[node]) for node in merchant_nodes]
            merchant_ranks.sort(key=lambda x: x[1], reverse=True)
            
            return merchant_ranks[:top_n]
        except Exception as e:
            logger.error(f"Error detecting top merchants: {e}")
            return []
    
    def detect_spending_clusters(self) -> List[Dict]:
        """Identify clusters of related spending using community detection."""
        try:
            # Convert to undirected for community detection
            G_undirected = self.G.to_undirected()
            
            # Detect communities using Louvain method
            communities = nx.community.louvain_communities(G_undirected, weight='weight')
            
            clusters = []
            for idx, community in enumerate(communities):
                # Get merchants and categories in this cluster
                merchants = [n for n in community if self.G.nodes[n].get('type') == 'merchant']
                categories = [n for n in community if self.G.nodes[n].get('type') == 'category']
                
                if merchants and categories:
                    total_spent = sum(self.G.nodes[m].get('total_spent', 0) for m in merchants)
                    clusters.append({
                        'cluster_id': idx,
                        'merchants': merchants[:3],  # Top 3
                        'categories': categories,
                        'total_spent': round(total_spent, 2),
                        'size': len(community)
                    })
            
            return sorted(clusters, key=lambda x: x['total_spent'], reverse=True)
        except Exception as e:
            logger.error(f"Error detecting clusters: {e}")
            return []
    
    def find_influential_categories(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find categories with highest centrality (influence on spending)."""
        try:
            # Calculate betweenness centrality
            centrality = nx.betweenness_centrality(self.G, weight='weight')
            
            # Filter categories
            category_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'category']
            category_centrality = [(node, centrality[node]) for node in category_nodes]
            category_centrality.sort(key=lambda x: x[1], reverse=True)
            
            return category_centrality[:top_n]
        except Exception as e:
            logger.error(f"Error finding influential categories: {e}")
            return []
    
    def detect_anomalous_connections(self) -> List[Dict]:
        """Find unusual merchant-category connections using statistical analysis."""
        anomalies = []
        
        try:
            # Calculate edge weights statistics
            edge_weights = [d['weight'] for u, v, d in self.G.edges(data=True)]
            if not edge_weights:
                return []
            
            mean_weight = sum(edge_weights) / len(edge_weights)
            std_weight = (sum((x - mean_weight) ** 2 for x in edge_weights) / len(edge_weights)) ** 0.5
            threshold = mean_weight + 2 * std_weight
            
            # Find edges exceeding threshold
            for u, v, data in self.G.edges(data=True):
                if data['weight'] > threshold:
                    u_type = self.G.nodes[u].get('type', 'unknown')
                    v_type = self.G.nodes[v].get('type', 'unknown')
                    
                    if u_type == 'merchant' and v_type == 'category':
                        anomalies.append({
                            'merchant': u,
                            'category': v,
                            'amount': round(data['weight'], 2),
                            'severity': 'high' if data['weight'] > threshold * 1.5 else 'moderate',
                            'reason': 'Unusually high spending concentration'
                        })
            
            return sorted(anomalies, key=lambda x: x['amount'], reverse=True)[:5]
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def get_spending_paths(self, category: str) -> List[str]:
        """Find all merchants connected to a specific category."""
        if category in self.G:
            return [n for n in self.G.predecessors(category) 
                   if self.G.nodes[n].get('type') == 'merchant']
        return []
    
    def generate_insights(self) -> GraphInsight:
        """Generate comprehensive insights from the graph."""
        top_merchants = self.detect_top_merchants()
        clusters = self.detect_spending_clusters()
        influential_cats = self.find_influential_categories()
        anomalies = self.detect_anomalous_connections()
        
        # Generate summary
        if top_merchants:
            top_name = top_merchants[0][0]
            summary = f"Your spending is heavily concentrated around '{top_name}'. "
        else:
            summary = "Insufficient data for graph analysis. "
        
        if clusters:
            summary += f"Identified {len(clusters)} spending clusters. "
        
        if anomalies:
            summary += f"Detected {len(anomalies)} anomalous spending patterns requiring attention."
        
        return GraphInsight(
            top_merchants=top_merchants,
            spending_clusters=clusters,
            influential_categories=influential_cats,
            anomalous_connections=anomalies,
            summary=summary
        )
    
    def export_for_visualization(self) -> Dict:
        """Export graph data in format suitable for D3.js visualization."""
        nodes = []
        links = []
        
        # Export nodes
        for node, data in self.G.nodes(data=True):
            nodes.append({
                'id': node,
                'type': data.get('type', 'unknown'),
                'value': data.get('total_spent', data.get('total_received', 1))
            })
        
        # Export edges
        for u, v, data in self.G.edges(data=True):
            links.append({
                'source': u,
                'target': v,
                'value': data.get('weight', 1)
            })
        
        return {'nodes': nodes, 'links': links}