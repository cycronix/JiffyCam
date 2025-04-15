"""
jiffyconfig.py: Configuration management for JiffyCam

This module handles loading, saving, and managing configuration settings
for the JiffyCam application.
"""

import os
import yaml
from collections import OrderedDict
from typing import Dict, Any

# Add a representer for OrderedDict to maintain order in YAML
yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))

# Add a constructor to load mappings as OrderedDict
yaml.add_constructor(yaml.resolver.Resolver.DEFAULT_MAPPING_TAG, 
                    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

# Common camera resolutions
RESOLUTIONS = {
    "4K (3840x2160)": (3840, 2160),
    "1080p (1920x1080)": (1920, 1080),
    "720p (1280x720)": (1280, 720),
    "480p (854x480)": (854, 480),
    "360p (640x360)": (640, 360),
    "Default (0x0)": (0, 0)
}

class JiffyConfig:
    def __init__(self, yaml_file: str = 'jiffycam.yaml'):
        """Initialize configuration manager.
        
        Args:
            yaml_file (str): Path to the YAML configuration file
        """
        self.yaml_file = yaml_file
        self.config = self.load_config()
        self.last_error = None

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file if it exists.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            'cam_device': '0',
            'session': 'Default',  # Default session value, but not saved to YAML anymore
            'cam_name': 'cam0',
            'resolution': '1920x1080',  # Combined resolution field
            'save_interval': 60,  # Changed to integer default
            'data_dir': 'JiffyData',  # Default data directory
            'dataserver_port': 8080,  # Default port for the JiffyCam data server
            'device_aliases': OrderedDict([   # Use OrderedDict for default device aliases
                ('USB0', '0'),
                ('USB1', '1'),
                ('Default', '0')
            ])
        }
        
        if os.path.exists(self.yaml_file):
            try:
                with open(self.yaml_file, 'r') as file:
                    config = yaml.safe_load(file)
                    
                    if config:
                        # Ensure save_interval is an integer
                        if 'save_interval' in config:
                            config['save_interval'] = int(config['save_interval'])
                        
                        # Ensure device_aliases exists and is an OrderedDict
                        if 'device_aliases' not in config:
                            config['device_aliases'] = default_config['device_aliases']
                        
                        # Handle legacy config with separate width and height
                        if 'cam_width' in config and 'cam_height' in config and 'resolution' not in config:
                            config['resolution'] = f"{config['cam_width']}x{config['cam_height']}"
                            # Remove old fields
                            config.pop('cam_width', None)
                            config.pop('cam_height', None)
                            
                        # Handle legacy cam_path key
                        if 'cam_path' in config and 'cam_device' not in config:
                            config['cam_device'] = config.pop('cam_path')
                            
                        return config
            except Exception as e:
                self.last_error = f"Error loading configuration: {str(e)}"
        return default_config

    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to YAML file.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary to save
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        print(f"Saving configuration to {self.yaml_file}")      # shouildnt happen?
        try:
            # Create a copy to avoid modifying the original
            config = config.copy()
            
            # Ensure data_dir is preserved if it exists in the current config
            if 'data_dir' not in config and hasattr(self, 'config') and isinstance(self.config, dict) and 'data_dir' in self.config:
                config['data_dir'] = self.config['data_dir']
            
            # Convert device_aliases to OrderedDict to preserve order
            if 'device_aliases' in config and isinstance(config['device_aliases'], dict):
                config['device_aliases'] = OrderedDict(config['device_aliases'])
            
            with open(self.yaml_file, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            return True
        except Exception as e:
            self.last_error = f"Failed to save configuration: {str(e)}"
            return False

    def get_resolution(self, resolution_name: str) -> tuple:
        """Get resolution dimensions from resolution name.
        
        Args:
            resolution_name (str): Name of the resolution
            
        Returns:
            tuple: (width, height) dimensions
        """
        return RESOLUTIONS.get(resolution_name, (0, 0))

    def get_resolution_name(self, width: int, height: int) -> str:
        """Get resolution name from dimensions.
        
        Args:
            width (int): Width in pixels
            height (int): Height in pixels
            
        Returns:
            str: Name of the resolution
        """
        for name, (w, h) in RESOLUTIONS.items():
            if w == width and h == height:
                return name
        return "Default (0x0)" 