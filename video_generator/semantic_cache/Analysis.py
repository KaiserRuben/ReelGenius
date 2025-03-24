#%% md
# # Embedding Model Comparison with Category-Based Analysis
# 
# This notebook analyzes embedding models' performance using real data from cache_keys.csv.
# Test cases are grouped by category, and similarity analysis is performed within each category.
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import networkx as nx

from video_generator.semantic_cache.model_loader import ModelLoader
#%% md
# ## Load and Process Data from CSV
#%%
def load_cache_data(filepath='../cache_keys.csv'):
    """
    Load and process the cache keys CSV file.
    Returns a dictionary of categories with their corresponding texts.
    """
    df = pd.read_csv(filepath, header=None,
                     names=['hash', 'category', 'use', 'content'])

    # Filter only entries marked as True in the 'use' column
    df = df[df['use'].astype(str).str.lower() == 'true']

    # Group by category
    category_groups = {}
    for category, group in df.groupby('category'):
        category_groups[category] = group['content'].tolist()

    return category_groups


# Load the categorized data
category_groups = load_cache_data()

print("Categories found:")
for category, texts in category_groups.items():
    print(f"{category}: {len(texts)} entries")
#%% md
# ## Display Text Mapping
# Show which text corresponds to each number in the visualizations
#%% raw
# from IPython.display import display, HTML
# 
# def create_text_mapping_table(category_groups):
#     html = """
#     <style>
#         .mapping-table {
#             border-collapse: collapse;
#             margin: 20px 0;
#             font-family: Arial, sans-serif;
#             width: 100%;
#         }
#         .mapping-table th {
#             background-color: #f8f9fa;
#             padding: 12px;
#             text-align: left;
#             border: 1px solid #dee2e6;
#         }
#         .mapping-table td {
#             padding: 12px;
#             border: 1px solid #dee2e6;
#         }
#         .mapping-table tr:nth-child(even) {
#             background-color: #f8f9fa;
#         }
#         .category-header {
#             background-color: #e9ecef;
#             font-weight: bold;
#             font-size: 1.1em;
#         }
#         .text-number {
#             font-weight: bold;
#             color: #495057;
#             width: 100px;
#         }
#     </style>
#     <h3>Text Mapping for Each Category</h3>
#     """
# 
#     for category, texts in category_groups.items():
#         html += f"""
#         <table class="mapping-table">
#             <tr class="category-header">
#                 <th colspan="2">{category}</th>
#             </tr>
#             <tr>
#                 <th>Reference</th>
#                 <th>Content</th>
#             </tr>
#         """
# 
#         for idx, text in enumerate(texts, 1):
#             html += f"""
#             <tr>
#                 <td class="text-number">Text {idx}</td>
#                 <td>{text}</td>
#             </tr>
#             """
# 
#         html += "</table>"
# 
#     return HTML(html)
# 
# # Display the formatted text mapping
# display(create_text_mapping_table(category_groups))
#%% md
# ## Define Models to Compare
#%%
models = {
    'all-MiniLM-L6-v2': ModelLoader('all-MiniLM-L6-v2'),
    'all-mpnet-base-v2': ModelLoader('all-mpnet-base-v2'),
    'multi-qa-mpnet-base-dot-v1': ModelLoader('multi-qa-mpnet-base-dot-v1'),
    'all-distilroberta-v1': ModelLoader('all-distilroberta-v1'),
    'intfloat/multilingual-e5-large-instruct': ModelLoader('intfloat/multilingual-e5-large-instruct')
}

#%%
thresholds = np.arange(0.1, 1.0, 0.02)
thresholds
#%% md
# ## Generate Embeddings by Category
#%%
def generate_embeddings(texts, model):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

# Generate embeddings for each category
category_embeddings = {}
for model_name, model in tqdm(models.items(), desc="Processing models"):
    category_embeddings[model_name] = {}
    for category, texts in category_groups.items():
        if texts:  # Only process categories with texts
            category_embeddings[model_name][category] = generate_embeddings(texts, model)
#%% md
# ## Analyze Similarities Within Categories
#%%
def create_combined_heatmaps(category_embeddings, category, texts, save_html=True,cutoff=0.7):
    """Create a combined heatmap visualization comparing all models for a category"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Count number of models for subplot layout
    n_models = len(category_embeddings)
    n_cols = 2  # We'll use 2 columns
    n_rows = (n_models + 1) // 2  # Calculate rows needed
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=list(category_embeddings.keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Create hover template using Plotly's syntax
    hover_template = (
        "Text: %{x} vs %{y}<br>" +
        "Similarity: %{z:.3f}<br><br>" +
        "Content 1: %{customdata[0]}<br>" +
        "Content 2: %{customdata[1]}"
    )
    
    # Add heatmaps for each model
    for idx, (model_name, model_categories) in enumerate(category_embeddings.items()):
        if category in model_categories:
            embeddings = model_categories[category]
            if len(embeddings) > 1:
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(embeddings)
                
                # Create text previews for hover data
                text_previews = [f"{text[:50]}..." if len(text) > 50 else text 
                               for text in texts]
                
                # Create hover data matrix as a 2D array matching the heatmap dimensions
                hover_data = [[
                    [text_previews[i], text_previews[j]]
                    for j in range(len(texts))
                ] for i in range(len(texts))]
                
                # Calculate subplot position
                row = (idx // 2) + 1
                col = (idx % 2) + 1
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=similarity_matrix,
                        x=[f"T{i+1}" for i in range(len(texts))],
                        y=[f"T{i+1}" for i in range(len(texts))],
                        customdata=hover_data,
                        hoverongaps=False,
                        hovertemplate=hover_template,
                        colorscale='RdBu_r',
                        zmin=0,
                        zmax=1
                    ),
                    row=row, col=col
                )
    
    # Update layout
    fig.update_layout(
        title=f'Model Comparison Heatmaps - Category: {category}',
        height=300 * n_rows,  # Adjust height based on number of rows
        width=1000,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, n_models + 1):
        fig.update_xaxes(title_text="Text Number", row=(i // 2) + 1, col=(i % 2) + 1)
        fig.update_yaxes(title_text="Text Number", row=(i // 2) + 1, col=(i % 2) + 1)
    
    if save_html:
        # Save the figure to an HTML file with plotly.js included
        fig.write_html(
            f'output/heatmap_visualization_{category}_cutoff_{cutoff:.1f}.html',
            include_plotlyjs=True,
            full_html=True
        )
    
    # fig.show()

# Generate combined heatmaps for each category
categories = set()
for model_embeddings in category_embeddings.values():
    categories.update(model_embeddings.keys())

for category in categories:
    print(f"\nGenerating combined heatmaps for category: {category}")
    for cutoff in thresholds:
        print(f"  Cutoff: {cutoff:.1f}")
        create_combined_heatmaps(category_embeddings, category, category_groups[category], 
                               cutoff=cutoff, save_html=True)
#%% md
# ## Clustering Analysis by Category
#%%
def cluster_texts(similarity_matrix, cutoff):
    """
    Creates a graph where nodes are text indices and an edge exists between two nodes
    if their similarity is above the cutoff. Returns the connected components as clusters.
    """
    n = similarity_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= cutoff:
                G.add_edge(i, j)
    return list(nx.connected_components(G))

def analyze_category_cutoffs(category_embeddings, thresholds):
    results = []
    for model_name, model_categories in category_embeddings.items():
        for category, embeddings in model_categories.items():
            if len(embeddings) > 1:  # Skip categories with single entry
                similarity_matrix = cosine_similarity(embeddings)
                for cutoff in thresholds:
                    clusters = cluster_texts(similarity_matrix, cutoff)
                    cluster_sizes = [len(c) for c in clusters]
                    results.append({
                        "model": model_name,
                        "category": category,
                        "cutoff": cutoff,
                        "num_clusters": len(clusters),
                        "min_cluster_size": min(cluster_sizes) if cluster_sizes else None,
                        "max_cluster_size": max(cluster_sizes) if cluster_sizes else None,
                        "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else None,
                        "clusters": clusters
                    })
    return pd.DataFrame(results)

# Define cutoff thresholds
cutoff_results = analyze_category_cutoffs(category_embeddings, thresholds)
#%% md
# ## Visualization: Number of Clusters vs. Cutoff Threshold by Category
#%%
# Plot clusters vs cutoff for each category and model
categories = cutoff_results['category'].unique()
num_categories = len(categories)
ncols = 2
nrows = (num_categories + 1) // 2

plt.figure(figsize=(15, 5 * nrows))
for idx, category in enumerate(categories, 1):
    plt.subplot(nrows, ncols, idx)
    category_data = cutoff_results[cutoff_results['category'] == category]

    for model_name in category_data['model'].unique():
        subset = category_data[category_data['model'] == model_name]
        plt.plot(subset['cutoff'], subset['num_clusters'],
                 marker='o', label=model_name)

    plt.xlabel("Cutoff Threshold")
    plt.ylabel("Number of Clusters")
    plt.title(f"Category: {category}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

# plt.show()
#%% md
# ## Network Graph Visualization by Category
#%%
def visualize_combined_clusters(category_embeddings, category, texts, cutoff=0.7, save_html=True):
    """Create a combined 3D visualization of clusters across different models with text preview on hover"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    model_colors = {
        'all-MiniLM-L6-v2': '#1f77b4',
        'all-mpnet-base-v2': '#ff7f0e',
        'multi-qa-mpnet-base-dot-v1': '#2ca02c',
        'all-distilroberta-v1': '#d62728',
        'intfloat/multilingual-e5-large-instruct': '#9467bd'
    }
    
    for model_name, model_categories in category_embeddings.items():
        if category in model_categories:
            embeddings = model_categories[category]
            
            if len(embeddings) > 1:
                similarity_matrix = cosine_similarity(embeddings)
                G = nx.Graph()
                edges = [(r, c) for r in range(len(embeddings)) 
                        for c in range(r + 1, len(embeddings)) 
                        if similarity_matrix[r, c] >= cutoff]
                G.add_edges_from(edges)
                
                pos = nx.spring_layout(G, dim=3, seed=42)
                
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_z = [pos[node][2] for node in G.nodes()]
                
                hover_texts = [
                    f'Text {i+1}: {text[:100]}{"..." if len(text) > 100 else ""}' 
                    for i, text in enumerate(texts)
                ]
                
                fig.add_trace(go.Scatter3d(
                    x=node_x,
                    y=node_y,
                    z=node_z,
                    mode='markers+text',
                    text=[f'T{i+1}' for i in range(len(embeddings))],
                    hovertext=hover_texts,
                    hoverinfo='text',
                    name=model_name,
                    marker=dict(
                        size=8,
                        color=model_colors[model_name],
                        symbol='circle'
                    )
                ))
                
                edge_x = []
                edge_y = []
                edge_z = []
                
                for edge in G.edges():
                    x0, y0, z0 = pos[edge[0]]
                    x1, y1, z1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])
                
                fig.add_trace(go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode='lines',
                    line=dict(color=model_colors[model_name], width=1),
                    hoverinfo='none',
                    showlegend=False
                ))
    
    fig.update_layout(
        title=f'Combined Cluster Visualization - Category: {category} (cutoff: {cutoff})',
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=800,
        height=800
    )
    
    if save_html:
        # Save the figure to an HTML file with plotly.js included
        fig.write_html(
            f'output/cluster_visualization_{category}_cutoff_{cutoff:.1f}.html',
            include_plotlyjs=True,
            full_html=True
        )
    # fig.show()

# Generate combined visualizations for each category and cutoff
for category in category_embeddings['all-MiniLM-L6-v2'].keys():
    if len(category_embeddings['all-MiniLM-L6-v2'][category]) > 1:
        print(f"\nVisualizing combined clusters for category: {category}")
        for cutoff in thresholds:
            print(f"  Cutoff: {cutoff:.1f}")
            visualize_combined_clusters(category_embeddings, category, category_groups[category], 
                                     cutoff=cutoff, save_html=True)
#%%
def visualize_threshold_gallery(category_embeddings, category, texts):
    """Create a threshold gallery visualization showing clusters at different thresholds"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Define thresholds with finer granularity
    my_thresholds = np.arange(0.5, 1.0, 0.05)
    # Calculate number of rows/columns for subplot layout
    n_thresholds = len(my_thresholds)
    n_cols = 3  # We'll use 3 columns
    n_rows = (n_thresholds + 2) // 3  # Calculate rows needed, rounding up
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'Threshold: {t:.2f}' for t in thresholds],
        specs=[[{'type': 'scene'} for _ in range(n_cols)] for _ in range(n_rows)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    model_colors = {
        'all-MiniLM-L6-v2': '#1f77b4',
        'all-mpnet-base-v2': '#ff7f0e',
        'multi-qa-mpnet-base-dot-v1': '#2ca02c',
        'all-distilroberta-v1': '#d62728',
        'intfloat/multilingual-e5-large-instruct': '#9467bd'
    }
    
    # Create visualizations for each threshold
    for idx, cutoff in enumerate(my_thresholds):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        for model_name, model_categories in category_embeddings.items():
            if category in model_categories:
                embeddings = model_categories[category]
                
                if len(embeddings) > 1:
                    similarity_matrix = cosine_similarity(embeddings)
                    G = nx.Graph()
                    edges = [(r, c) for r in range(len(embeddings)) 
                            for c in range(r + 1, len(embeddings)) 
                            if similarity_matrix[r, c] >= cutoff]
                    G.add_edges_from(edges)
                    
                    pos = nx.spring_layout(G, dim=3, seed=42)
                    
                    node_x = [pos[node][0] for node in G.nodes()]
                    node_y = [pos[node][1] for node in G.nodes()]
                    node_z = [pos[node][2] for node in G.nodes()]
                    
                    hover_texts = [
                        f'Text {i+1}: {text[:100]}{"..." if len(text) > 100 else ""}' 
                        for i, text in enumerate(texts)
                    ]
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=node_x,
                            y=node_y,
                            z=node_z,
                            mode='markers+text',
                            text=[f'T{i+1}' for i in range(len(embeddings))],
                            hovertext=hover_texts,
                            hoverinfo='text',
                            name=model_name,
                            marker=dict(
                                size=8,
                                color=model_colors[model_name],
                                symbol='circle'
                            ),
                            showlegend=idx == 0  # Only show legend for first subplot
                        ),
                        row=row, col=col
                    )
                    
                    edge_x = []
                    edge_y = []
                    edge_z = []
                    
                    for edge in G.edges():
                        x0, y0, z0 = pos[edge[0]]
                        x1, y1, z1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_z.extend([z0, z1, None])
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=edge_x,
                            y=edge_y,
                            z=edge_z,
                            mode='lines',
                            line=dict(color=model_colors[model_name], width=1),
                            hoverinfo='none',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
        
        # Update subplot layout
        fig.update_scenes(
            dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            row=row, col=col
        )
    
    # Update overall layout
    fig.update_layout(
        title=f'Cluster Evolution Gallery - Category: {category}',
        height=400 * n_rows,
        width=1200,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save to HTML
    fig.write_html(
        f'output/cluster_gallery_{category}.html',
        include_plotlyjs=True,
        full_html=True
    )
    
    # Show in notebook
    # fig.show()

def create_heatmap_gallery(category_embeddings, category, texts):
    """Create a heatmap gallery showing similarity matrices at different thresholds"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Define thresholds with finer granularity    
    # Calculate layout
    n_models = len(category_embeddings)
    n_cols = 2  # 2 columns for models
    n_rows = n_models  # One row per model
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'{model} (Threshold Gallery)' for model in category_embeddings.keys()],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    # Create hover template
    hover_template = (
        "Text: %{x} vs %{y}<br>" +
        "Similarity: %{z:.3f}<br><br>" +
        "Content 1: %{customdata[0]}<br>" +
        "Content 2: %{customdata[1]}"
    )
    
    # Add heatmaps for each model
    for idx, (model_name, model_categories) in enumerate(category_embeddings.items()):
        if category in model_categories:
            embeddings = model_categories[category]
            if len(embeddings) > 1:
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(embeddings)
                
                # Create text previews for hover data
                text_previews = [f"{text[:50]}..." if len(text) > 50 else text 
                               for text in texts]
                
                # Create hover data matrix
                hover_data = [[
                    [text_previews[i], text_previews[j]]
                    for j in range(len(texts))
                ] for i in range(len(texts))]
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=similarity_matrix,
                        x=[f"T{i+1}" for i in range(len(texts))],
                        y=[f"T{i+1}" for i in range(len(texts))],
                        customdata=hover_data,
                        hoverongaps=False,
                        hovertemplate=hover_template,
                        colorscale='RdBu_r',
                        zmin=0,
                        zmax=1,
                        showscale=True,
                        name=model_name
                    ),
                    row=idx+1, col=1
                )
    
    # Update layout
    fig.update_layout(
        title=f'Similarity Heatmap Gallery - Category: {category}',
        height=300 * n_rows,
        width=1200,
        showlegend=False
    )
    
    # Save to HTML
    fig.write_html(
        f'output/heatmap_gallery_{category}.html',
        include_plotlyjs=True,
        full_html=True
    )
    
    # Show in notebook
    # fig.show()

# Generate galleries for each category
for category in category_embeddings['all-MiniLM-L6-v2'].keys():
    if len(category_embeddings['all-MiniLM-L6-v2'][category]) > 1:
        print(f"\nGenerating galleries for category: {category}")
        visualize_threshold_gallery(category_embeddings, category, category_groups[category])
        create_heatmap_gallery(category_embeddings, category, category_groups[category])
#%% md
# # Slider
#%%
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools


def create_slider_visualization(category_embeddings, category, texts):
    """Create an interactive visualization with threshold slider"""
    fig = go.Figure()
    active_models = set()
    model_colors = {
        'all-MiniLM-L6-v2': '#1f77b4',
        'all-mpnet-base-v2': '#ff7f0e',
        'multi-qa-mpnet-base-dot-v1': '#2ca02c',
        'all-distilroberta-v1': '#d62728',
        'intfloat/multilingual-e5-large-instruct': '#9467bd'
    }

    # Pre-calculate layouts and similarities
    initial_layouts = {}
    similarity_matrices = {}

    # Calculate layouts with similarity-weighted edges
    for idx, (model_name, model_categories) in enumerate(category_embeddings.items()):
        if category in model_categories:
            embeddings = model_categories[category]
            if len(embeddings) > 1:
                # Calculate similarity once
                similarity_matrix = cosine_similarity(embeddings)
                similarity_matrices[model_name] = similarity_matrix

                # Create weighted graph for layout
                G_full = nx.Graph()
                G_full.add_nodes_from(range(len(embeddings)))

                # Add edges with similarity weights
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        G_full.add_edge(i, j, weight=similarity_matrix[i, j])

                # Use different seeds for different models
                initial_layouts[model_name] = nx.spring_layout(
                    G_full,
                    dim=3,
                    seed=42 + idx,  # Different seed per model
                    weight='weight',  # Use similarity weights
                    k=2 / np.sqrt(len(embeddings))  # Optimal distance between nodes
                )

    # Define thresholds for the slider
    thresholds = np.linspace(0, 1, 20)
    frames = []

    # Create frames for different thresholds
    for threshold in thresholds:
        frame_data = []
        for model_name in similarity_matrices.keys():
            similarity_matrix = similarity_matrices[model_name]
            pos = initial_layouts[model_name]

            # Get connected components at this threshold
            edges = [(r, c) for r, c in itertools.combinations(range(len(similarity_matrix)), 2)
                     if similarity_matrix[r, c] >= threshold]

            # Add nodes unconditionally
            node_x = [pos[i][0] for i in range(len(similarity_matrix))]
            node_y = [pos[i][1] for i in range(len(similarity_matrix))]
            node_z = [pos[i][2] for i in range(len(similarity_matrix))]

            hover_texts = [
                f'Text {i + 1}: {texts[i][:100]}{"..." if len(texts[i]) > 100 else ""}'
                for i in range(len(similarity_matrix))
            ]

            active_models.add(model_name)

            # Always show legend for nodes
            frame_data.append(
                go.Scatter3d(
                    x=node_x,
                    y=node_y,
                    z=node_z,
                    mode='markers+text',
                    text=[f'T{i + 1}' for i in range(len(similarity_matrix))],
                    hovertext=hover_texts,
                    hoverinfo='text',
                    name=model_name,
                    marker=dict(
                        size=8,
                        color=model_colors[model_name],
                        symbol='circle'
                    ),
                    showlegend=True,  # Always show in legend
                    legendgroup=model_name,  # Group traces by model
                    legendgrouptitle_text=model_name if float(threshold) == float(thresholds[0]) else None
                )
            )

            # Only add edges if they exist
            if edges:
                edge_x = []
                edge_y = []
                edge_z = []
                for edge in edges:
                    x0, y0, z0 = pos[edge[0]]
                    x1, y1, z1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])

                frame_data.append(
                    go.Scatter3d(
                        x=edge_x,
                        y=edge_y,
                        z=edge_z,
                        mode='lines',
                        line=dict(color=model_colors[model_name], width=1),
                        hoverinfo='none',
                        showlegend=False,
                        name=f"{model_name}_edges"
                    )
                )

        frames.append(go.Frame(data=frame_data, name=f"{threshold:.2f}"))

    # Set up the initial state
    fig.frames = frames
    fig.add_traces(frames[0].data)

    # Update layout
    fig.update_layout(
        title=f'Interactive Cluster Visualization - Category: {category} ({len(active_models)} models)',
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=500, redraw=True),
                        fromcurrent=True,
                        transition=dict(
                            duration=300,
                            easing="quadratic-in-out"
                        )
                    )]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0)
                    )]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=16),
                prefix="Threshold: ",
                visible=True,
                xanchor="right"
            ),
            pad=dict(t=50, b=10),
            len=0.9,
            x=0.1,
            y=0,
            steps=[dict(
                args=[[f"{threshold:.2f}"],
                      dict(frame=dict(duration=300, redraw=True),
                           mode="immediate",
                           transition=dict(duration=300))
                      ],
                label=f"{threshold:.2f}",
                method="animate"
            ) for threshold in thresholds]
        )]
    )

    return fig
# Test the visualization
for category in category_embeddings['all-MiniLM-L6-v2'].keys():
    if len(category_embeddings['all-MiniLM-L6-v2'][category]) > 1:
        print(f"\nGenerating interactive visualization for category: {category}")
        fig = create_slider_visualization(category_embeddings, category, category_groups[category])
        
    
        # Save to HTML
        fig.write_html(
            f'output/interactive_clusters_{category}.html',
            include_plotlyjs=True,
            full_html=True
        )
#%%
def create_professional_index(categories, category_groups, cutoff_results):
    """Create a professional index page with gallery and individual threshold views"""
    with open('index.html', 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>Embedding Model Analysis</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                :root {{
                    --primary-color: #0066cc;
                    --secondary-color: #f8f9fa;
                    --border-color: #e5e5e7;
                    --text-color: #1d1d1f;
                    --radius: 12px;
                }}
                
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: #f5f5f7;
                    color: var(--text-color);
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 2rem;
                }}
                
                .header {{
                    background: white;
                    padding: 2rem;
                    border-radius: var(--radius);
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    margin-bottom: 2rem;
                }}
                
                .header h1 {{
                    margin: 0 0 1rem 0;
                    font-weight: 600;
                    font-size: 2rem;
                    color: var(--text-color);
                }}
                
                .model-tags {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                    margin-top: 1rem;
                }}
                
                .model-tag {{
                    background: var(--secondary-color);
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    color: var(--text-color);
                }}
                
                .overview-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }}
                
                .stats-card {{
                    background: white;
                    padding: 1.5rem;
                    border-radius: var(--radius);
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                
                .category-section {{
                    background: white;
                    border-radius: var(--radius);
                    padding: 2rem;
                    margin-bottom: 2rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                
                .category-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1.5rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid var(--border-color);
                }}
                
                .gallery-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }}
                
                @media (max-width: 768px) {{
                    .gallery-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
                
                .gallery-card {{
                    background: var(--secondary-color);
                    border-radius: var(--radius);
                    overflow: hidden;
                    transition: transform 0.2s;
                }}
                
                .gallery-card:hover {{
                    transform: translateY(-3px);
                }}
                
                .gallery-content {{
                    padding: 1.5rem;
                }}
                
                .view-button {{
                    display: inline-block;
                    background: var(--primary-color);
                    color: white;
                    text-decoration: none;
                    padding: 0.8rem 1.5rem;
                    border-radius: 25px;
                    font-weight: 500;
                    transition: background-color 0.2s;
                }}
                
                .view-button:hover {{
                    background: #0077ed;
                }}
                
                .threshold-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
                    gap: 1rem;
                    margin-top: 1.5rem;
                    padding: 1.5rem;
                    background: var(--secondary-color);
                    border-radius: var(--radius);
                }}
                
                .threshold-link {{
                    display: block;
                    padding: 0.5rem;
                    text-align: center;
                    background: white;
                    border-radius: 6px;
                    color: var(--text-color);
                    text-decoration: none;
                    font-size: 0.9rem;
                    transition: background-color 0.2s;
                }}
                
                .threshold-link:hover {{
                    background: #e9ecef;
                }}
                
                .section-title {{
                    font-size: 1.1rem;
                    font-weight: 500;
                    margin: 1rem 0;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Embedding Model Analysis</h1>
                    <div class="model-tags">
                        <span class="model-tag">all-MiniLM-L6-v2</span>
                        <span class="model-tag">all-mpnet-base-v2</span>
                        <span class="model-tag">multi-qa-mpnet-base-dot-v1</span>
                        <span class="model-tag">all-distilroberta-v1</span>
                        <span class="model-tag">multilingual-e5-large-instruct</span>
                    </div>
                </div>

                <div class="overview-grid">
                    <div class="stats-card">
                        <h3>📊 Analysis Overview</h3>
                        <p>Categories analyzed: {len(categories)}</p>
                        <p>Total texts analyzed: {sum(len(texts) for texts in category_groups.values())}</p>
                        <p>Similarity thresholds: 0.50 - 0.95 (step: 0.05)</p>
                    </div>
                    <div class="stats-card">
                        <h3>🎯 Visualization Types</h3>
                        <ul>
                            <li>Gallery View: Evolution across thresholds</li>
                            <li>Individual Threshold Views: Detailed analysis</li>
                            <li>Heatmaps: Pairwise similarity analysis</li>
                        </ul>
                    </div>
                </div>
        """)

        # Add section for each category
        for category in categories:
            f.write(f"""
                <div class="category-section">
                    <div class="category-header">
                        <h2>{category}</h2>
                        <div>{len(category_groups[category])} texts analyzed</div>
                    </div>

                    <div class="gallery-grid">
                        <div class="gallery-card">
                            <div class="gallery-content">
                                <h3>Cluster Evolution Gallery</h3>
                                <p>Compare how clusters evolve across different similarity thresholds (0.50 - 0.95)</p>
                                <a href="output/cluster_gallery_{category}.html" class="view-button">View Gallery</a>
                            </div>
                        </div>
                        <div class="gallery-card">
                            <div class="gallery-content">
                                <h3>Heatmap Gallery</h3>
                                <p>Analyze pairwise similarities between texts across different models</p>
                                <a href="output/heatmap_gallery_{category}.html" class="view-button">View Gallery</a>
                            </div>
                        </div>
            """)
            f.write(f"""
        <div class="gallery-card">
            <div class="gallery-content">
                <h3>Interactive Threshold Explorer</h3>
                <p>Explore how text relationships evolve as similarity threshold changes:</p>
                <ul>
                    <li>Use the slider to adjust threshold (0.50 - 0.95)</li>
                    <li>Watch points and connections update in real-time</li>
                    <li>Play animation to see smooth transitions</li>
                    <li>Hover over points to see text content</li>
                </ul>
                <a href="output/interactive_clusters_{category}.html" class="view-button">Launch Interactive View</a>
            </div>
        </div>
                    </div>

                    <div class="section-title">Individual Threshold Views</div>
                    <div class="threshold-grid">
    """)

            # Add links for individual threshold visualizations
            for cutoff in thresholds:
                f.write(f"""
                    <a href="output/cluster_visualization_{category}_cutoff_{cutoff:.1f}.html" 
                       class="threshold-link">
                        Clusters {cutoff:.1f}
                    </a>
                    <a href="output/heatmap_visualization_{category}_cutoff_{cutoff:.1f}.html"
                       class="threshold-link">
                        Heatmap {cutoff:.1f}
                    </a>
                """)

            f.write("""
                    </div>
                </div>
            """)

        f.write("""
            </div>
        </body>
        </html>
        """)

# Create the main index file
create_professional_index(categories, category_groups, cutoff_results)
#%% md
# # Debug
#%%
def debug_similarities(category_embeddings, category):
    """Debug function to analyze similarity distributions across models"""
    print(f"\nDebugging similarities for category: {category}")
    
    for model_name, model_categories in category_embeddings.items():
        if category in model_categories:
            embeddings = model_categories[category]
            if len(embeddings) > 1:
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(embeddings)
                
                # Get statistics
                sim_mean = np.mean(similarity_matrix)
                sim_std = np.std(similarity_matrix)
                sim_min = np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                sim_max = np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                
                # Count connections at different thresholds
                connections = {}
                for threshold in thresholds:
                    connections[threshold] = np.sum(similarity_matrix >= threshold) // 2  # Divide by 2 because matrix is symmetric
                
                print(f"\nModel: {model_name}")
                print(f"Similarity stats:")
                print(f"  Mean: {sim_mean:.3f}")
                print(f"  Std:  {sim_std:.3f}")
                print(f"  Min:  {sim_min:.3f}")
                print(f"  Max:  {sim_max:.3f}")
                print("\nConnections at thresholds:")
                for threshold, count in connections.items():
                    print(f"  {threshold:.1f}: {count} connections")
                
                # Print raw similarity matrix for small number of texts
                if len(embeddings) <= 5:
                    print("\nRaw similarity matrix:")
                    print(similarity_matrix)

# Test for a specific category
for category in category_embeddings['all-MiniLM-L6-v2'].keys():
    debug_similarities(category_embeddings, category)
#%%
def debug_threshold_points(category_embeddings, category):
    """Debug the number of points shown at each threshold"""
    
    for model_name, model_categories in category_embeddings.items():
        print(f"\nModel: {model_name}")
        print("Threshold | Points | Connections")
        print("-" * 35)
        
        if category in model_categories:
            embeddings = model_categories[category]
            if len(embeddings) > 1:
                similarity_matrix = cosine_similarity(embeddings)
                
                for threshold in thresholds:
                    # Get points that have at least one connection at this threshold
                    connections = similarity_matrix >= threshold
                    points_with_connections = set()
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            if connections[i, j]:
                                points_with_connections.add(i)
                                points_with_connections.add(j)
                    
                    n_connections = np.sum(connections) // 2  # Divide by 2 because matrix is symmetric
                    print(f"{threshold:9.2f} | {len(points_with_connections):6d} | {n_connections:6d}")

# Test for a specific category
for category in category_embeddings['all-MiniLM-L6-v2'].keys():
    print(f"\nCategory: {category}")
    debug_threshold_points(category_embeddings, category)
#%%
