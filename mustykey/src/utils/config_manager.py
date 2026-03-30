# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 09:21:00 2025

@author: shnoz
"""

# -------------------------------------------------------------
# File: src/utils/config_manager.py
# YAML configuration loader with validation.
# -------------------------------------------------------------

import yaml
import os


class ConfigManager:
    @staticmethod
    def load(path):

        # 1. Check file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        # 2. Load YAML
        with open(path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError("YAML file is empty or invalid.")

        # 3. Validate required fields
        required = ["N", "L", "dt", "T"]
        for key in required:
            if key not in config:
                raise KeyError(f"Missing required config key: '{key}'")

        return config
