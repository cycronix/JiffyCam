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
    def __init__(self, yaml_file: str = 'jiffycam.yaml', session: str = None, data_dir: str = 'JiffyData', require_config_exists: bool = False):
        """Initialize configuration manager.
        
        Args:
            yaml_file (str): Base name of the YAML configuration file (default: 'jiffycam.yaml')
            session (str): Session name to use for configuration lookup
            data_dir (str): Base directory for session data (default: 'JiffyData')
            require_config_exists (bool): If True, raise FileNotFoundError when config file doesn't exist
        """
        self.base_yaml_file = yaml_file
        self.data_dir = data_dir
        self.session = session
        self.require_config_exists = require_config_exists
        self.yaml_file = self._get_config_path(session)
        self.config = self.load_config()
        self.last_error = None

    def _get_config_path(self, session: str = None) -> str:
        """Get the path to the configuration file based on session.
        
        If a session is provided, construct the path to the session-specific
        configuration file. If no session is provided, use the default base file.
        
        Args:
            session (str, optional): Session name
            
        Returns:
            str: Path to the configuration file
        """
        # The data_dir already contains the session name, so we don't need to add it again
        return os.path.join(self.data_dir, self.base_yaml_file)

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file if it exists.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If require_config_exists is True and the specified config file doesn't exist
            ValueError: If the config file exists but can't be loaded
        """
        default_config = {
            'cam_device': '0',
            'session': 'Default',  # Default session value, but not saved to YAML anymore
            'cam_name': 'cam0',
            'resolution': '1920x1080',  # Combined resolution field
            'save_interval': 60,  # Changed to integer default
            'detect_interval': 5,  # Changed to integer default
            'data_dir': self.data_dir,  # Default data directory
            'dataserver_port': 8080,  # Default port for the JiffyCam data server
        }
        
        # Check if we should enforce the config file exists
        if not os.path.exists(self.yaml_file):
            # Only raise an error if we're requiring the config file to exist
            if self.require_config_exists:
                raise FileNotFoundError(f"Configuration file not found: {self.yaml_file}")
            return default_config
        
        try:
            with open(self.yaml_file, 'r') as file:
                config = yaml.safe_load(file)
                
                if config is None:
                    raise ValueError(f"Empty or invalid YAML in {self.yaml_file}")
                    
                # Ensure save_interval is an integer
                if 'save_interval' in config:
                    config['save_interval'] = int(config['save_interval'])
                
                if 'detect_interval' in config:
                    config['detect_interval'] = int(config['detect_interval'])
                
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
        except yaml.YAMLError as e:
            error_msg = f"Error parsing YAML in {self.yaml_file}: {str(e)}"
            self.last_error = error_msg
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}"
            self.last_error = error_msg
            raise ValueError(error_msg)

    def save_config(self, config: Dict[str, Any], session: str = None) -> bool:
        """Save configuration to YAML file.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary to save
            session (str, optional): Session name to use for saving
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Create a copy to avoid modifying the original
            config = config.copy()
            
            # Ensure data_dir is preserved if it exists in the current config
            if 'data_dir' not in config and hasattr(self, 'config') and isinstance(self.config, dict) and 'data_dir' in self.config:
                config['data_dir'] = self.config['data_dir']
            
            # If a new session is provided, update the yaml_file path
            if session and session != self.session:
                self.session = session
                self.yaml_file = self._get_config_path(session)
            
            # If we're using a session-specific config path, ensure the directory exists
            if self.session:
                session_dir = os.path.dirname(self.yaml_file)
                if not os.path.exists(session_dir):
                    os.makedirs(session_dir, exist_ok=True)
            
            with open(self.yaml_file, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            print(f"Saved configuration to: {self.yaml_file}")
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