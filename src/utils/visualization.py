import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import os

class DataVisualizer:
    """
    Class for visualizing data and model results for agroforestry land suitability
    and GHG emission reduction prediction.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the DataVisualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def plot_feature_distributions(self, 
                                  data: pd.DataFrame, 
                                  features: List[str] = None,
                                  n_cols: int = 3,
                                  figsize: Tuple[int, int] = (18, 12),
                                  save_path: str = None) -> plt.Figure:
        """
        Plot distributions of features.
        
        Args:
            data: DataFrame containing the data
            features: List of features to plot (if None, plot all numeric features)
            n_cols: Number of columns in the subplot grid
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if features is None:
            features = data.select_dtypes(include=['number']).columns.tolist()
        
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if i < len(axes):
                sns.histplot(data[feature], kde=True, ax=axes[i])
                axes[i].set_title(feature)
        
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, 
                               data: pd.DataFrame,
                               features: List[str] = None,
                               figsize: Tuple[int, int] = (12, 10),
                               save_path: str = None) -> plt.Figure:
        """
        Plot correlation matrix of features.
        
        Args:
            data: DataFrame containing the data
            features: List of features to include (if None, use all numeric features)
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if features is None:
            numeric_data = data.select_dtypes(include=['number'])
        else:
            numeric_data = data[features]
        
        corr = numeric_data.corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, 
                   fmt='.2f', square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_target_distribution(self, 
                                data: pd.DataFrame,
                                target_column: str,
                                title: str = None,
                                figsize: Tuple[int, int] = (10, 6),
                                save_path: str = None) -> plt.Figure:
        """
        Plot distribution of the target variable.
        
        Args:
            data: DataFrame containing the data
            target_column: Name of the target column
            title: Title of the plot
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        if data[target_column].dtype in ['int64', 'int32', 'bool']:
            sns.countplot(x=target_column, data=data)
            plt.title(title or f'Distribution of {target_column}')
            plt.xlabel(target_column)
            plt.ylabel('Count')
        else:
            sns.histplot(data[target_column], kde=True)
            plt.title(title or f'Distribution of {target_column}')
            plt.xlabel(target_column)
            plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_feature_vs_target(self, 
                              data: pd.DataFrame,
                              feature: str,
                              target: str,
                              hue: str = None,
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: str = None) -> plt.Figure:
        """
        Plot relationship between a feature and the target.
        
        Args:
            data: DataFrame containing the data
            feature: Name of the feature
            target: Name of the target
            hue: Name of the column to use for color encoding
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        if data[target].dtype in ['int64', 'int32', 'bool']:
            sns.boxplot(x=target, y=feature, data=data, hue=hue)
            plt.title(f'{feature} vs {target}')
        else:
            sns.scatterplot(x=feature, y=target, data=data, hue=hue)
            plt.title(f'{feature} vs {target}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_spatial_data(self, 
                         gdf: Any,  # GeoDataFrame from geopandas
                         column: str,
                         title: str = None,
                         cmap: str = 'viridis',
                         figsize: Tuple[int, int] = (12, 10),
                         save_path: str = None) -> plt.Figure:
        """
        Plot spatial data using GeoPandas.
        
        Args:
            gdf: GeoDataFrame containing the spatial data
            column: Column to use for coloring
            title: Title of the plot
            cmap: Colormap to use
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required for spatial plotting")
        
        fig, ax = plt.subplots(figsize=figsize)
        gdf.plot(column=column, cmap=cmap, legend=True, ax=ax)
        ax.set_title(title or f'Spatial Distribution of {column}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def create_interactive_map(self, 
                              gdf: Any,  # GeoDataFrame from geopandas
                              column: str,
                              popup_columns: List[str] = None,
                              title: str = None,
                              save_path: str = None) -> Any:  # folium.Map
        """
        Create an interactive map using Folium.
        
        Args:
            gdf: GeoDataFrame containing the spatial data
            column: Column to use for coloring
            popup_columns: Columns to include in popups
            title: Title of the map
            save_path: Path to save the map HTML
            
        Returns:
            Folium Map object
        """
        try:
            import folium
            from folium.features import GeoJsonPopup
        except ImportError:
            raise ImportError("folium is required for interactive maps")
        
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        
        center = [gdf_wgs84.geometry.centroid.y.mean(), gdf_wgs84.geometry.centroid.x.mean()]
        
        m = folium.Map(location=center, zoom_start=10)
        
        if title:
            title_html = f'<h3 align="center">{title}</h3>'
            m.get_root().html.add_child(folium.Element(title_html))
        
        if popup_columns:
            popup = GeoJsonPopup(
                fields=popup_columns,
                aliases=popup_columns,
                localize=True,
                labels=True
            )
        else:
            popup = None
        
        folium.Choropleth(
            geo_data=gdf_wgs84.__geo_interface__,
            name=column,
            data=gdf_wgs84,
            columns=['id', column],
            key_on='feature.id',
            fill_color='YlGnBu',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=column
        ).add_to(m)
        
        folium.GeoJson(
            gdf_wgs84,
            name='geojson',
            style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
            popup=popup
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
        
        return m
