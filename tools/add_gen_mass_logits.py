#!/usr/bin/env python3
"""
Script to add gen_mass_logits to h5 files based on the mass of the sample.

This script reads an h5 file, finds the mass information, calculates gen_mass_logits
based on which mass category the sample belongs to, and saves it back to the h5 file.

Mass classes: 200, 300, 400, 500, 600, 700, 800, 900, 1000
"""

import h5py
import numpy as np
import argparse
import sys
from pathlib import Path


def find_mass_key(h5_file):
    """Find the key that contains mass information in the h5 file."""
    possible_keys = [
        'REGRESSIONS/EVENT/gen_mass',  # Common in combined training files
        'TARGETS/gen_mass',  # Common in test files
        'EVENT/gen_mass',
        'EVENT/mass',
        'EVENT/gen_mass_true',
        'EVENT/gen_mass_gen',
        'EVENT/resonance_mass',
        'gen_mass',
        'mass',
    ]
    
    for key in possible_keys:
        if key in h5_file:
            return key
    
    # Try to find any key containing 'mass' in REGRESSIONS/EVENT group
    if 'REGRESSIONS' in h5_file and 'EVENT' in h5_file['REGRESSIONS']:
        for key in h5_file['REGRESSIONS']['EVENT'].keys():
            if 'mass' in key.lower():
                return f'REGRESSIONS/EVENT/{key}'
    
    # Try to find any key containing 'mass' in EVENT group
    if 'EVENT' in h5_file:
        for key in h5_file['EVENT'].keys():
            if 'mass' in key.lower():
                return f'EVENT/{key}'
    
    # Try to find gen_mass in TARGETS
    if 'TARGETS' in h5_file:
        for key in h5_file['TARGETS'].keys():
            if 'mass' in key.lower():
                return f'TARGETS/{key}'
    
    return None


def calculate_gen_mass_logits(mass_values, mass_classes, return_class_indices=False):
    """
    Calculate gen_mass_logits based on mass values and mass classes.
    
    Args:
        mass_values: Array of mass values
        mass_classes: List of mass class values [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        return_class_indices: If True, return 1D class indices instead of 2D logits
    
    Returns:
        If return_class_indices=False: Array of shape (n_samples, n_classes) with logits
        If return_class_indices=True: Array of shape (n_samples,) with class indices
    """
    n_samples = len(mass_values)
    n_classes = len(mass_classes)
    mass_classes_array = np.array(mass_classes, dtype=np.float32)
    
    # For each sample, find which mass class it belongs to
    mass_values_float = mass_values.astype(np.float32)
    
    # Find the closest mass class for each sample
    # Using broadcasting to compute distances
    distances = np.abs(mass_values_float[:, np.newaxis] - mass_classes_array[np.newaxis, :])
    closest_indices = np.argmin(distances, axis=1)
    
    if return_class_indices:
        # Return 1D class indices
        return closest_indices.astype(np.int64)
    
    # Initialize logits array
    logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    
    # Set high logit value for the correct class
    # Using 10.0 ensures that after softmax, the correct class will have probability ~1.0
    for i, idx in enumerate(closest_indices):
        logits[i, idx] = 10.0
    
    return logits


def add_gen_mass_logits(h5_path, mass_classes=None, output_path=None, inplace=True):
    """
    Add gen_mass_logits to an h5 file.
    
    Args:
        h5_path: Path to the input h5 file
        mass_classes: List of mass class values (default: [200, 300, 400, 500, 600, 700, 800, 900, 1000])
        output_path: Path to save the output file (if None and inplace=False, adds _with_logits suffix)
        inplace: If True, modify the file in place
    """
    if mass_classes is None:
        mass_classes = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: File {h5_path} does not exist", file=sys.stderr)
        return False
    
    print(f"Processing {h5_path}...")
    
    # Determine output path
    if inplace:
        output_path = h5_path
        temp_path = h5_path.with_suffix('.tmp.h5')
    else:
        if output_path is None:
            output_path = h5_path.with_name(h5_path.stem + '_with_logits.h5')
        else:
            output_path = Path(output_path)
        temp_path = output_path
    
    try:
        # Open the h5 file
        with h5py.File(h5_path, 'r') as f_in:
            # Find the mass key
            mass_key = find_mass_key(f_in)
            if mass_key is None:
                print(f"Error: Could not find mass information in {h5_path}", file=sys.stderr)
                print(f"Available keys in EVENT group: {list(f_in.get('EVENT', {}).keys())}", file=sys.stderr)
                return False
            
            print(f"Found mass key: {mass_key}")
            mass_data = f_in[mass_key][:]
            print(f"Mass data shape: {mass_data.shape}")
            print(f"Mass range: [{mass_data.min():.2f}, {mass_data.max():.2f}]")
            
            # Calculate gen_mass_logits
            gen_mass_logits = calculate_gen_mass_logits(mass_data, mass_classes)
            print(f"Generated gen_mass_logits shape: {gen_mass_logits.shape}")
            print(f"gen_mass_logits range: [{gen_mass_logits.min():.2f}, {gen_mass_logits.max():.2f}]")
            
            # Create output file
            with h5py.File(temp_path, 'w') as f_out:
                # Copy all groups and datasets
                def copy_item(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # Copy dataset
                        if name == 'EVENT/gen_mass_logits':
                            # Skip if it already exists (we'll add it later)
                            return
                        # Preserve compression and other attributes
                        compression = obj.compression if hasattr(obj, 'compression') else None
                        compression_opts = obj.compression_opts if hasattr(obj, 'compression_opts') else None
                        f_out.create_dataset(
                            name, 
                            data=obj[:], 
                            compression=compression,
                            compression_opts=compression_opts,
                            shuffle=obj.shuffle if hasattr(obj, 'shuffle') else False
                        )
                        # Copy attributes
                        for attr_name, attr_value in obj.attrs.items():
                            f_out[name].attrs[attr_name] = attr_value
                    elif isinstance(obj, h5py.Group):
                        # Create group
                        if name not in f_out:
                            f_out.create_group(name)
                        # Copy group attributes
                        for attr_name, attr_value in obj.attrs.items():
                            f_out[name].attrs[attr_name] = attr_value
                
                f_in.visititems(copy_item)
                
                # Add gen_mass_logits to REGRESSIONS/EVENT group (where regression targets are loaded from)
                # Create REGRESSIONS/EVENT if it doesn't exist
                if 'REGRESSIONS' not in f_out:
                    f_out.create_group('REGRESSIONS')
                if 'EVENT' not in f_out['REGRESSIONS']:
                    f_out['REGRESSIONS'].create_group('EVENT')
                
                # Check if gen_mass_logits already exists in REGRESSIONS/EVENT
                if 'REGRESSIONS/EVENT/gen_mass_logits' in f_out:
                    print("Warning: REGRESSIONS/EVENT/gen_mass_logits already exists, overwriting...")
                    del f_out['REGRESSIONS/EVENT/gen_mass_logits']
                
                f_out.create_dataset(
                    'REGRESSIONS/EVENT/gen_mass_logits', 
                    data=gen_mass_logits, 
                    compression='gzip',
                    shuffle=True
                )
                print(f"Added REGRESSIONS/EVENT/gen_mass_logits to {temp_path}")
                
                # Also add to EVENT group for compatibility
                if 'EVENT' not in f_out:
                    f_out.create_group('EVENT')
                
                if 'EVENT/gen_mass_logits' in f_out:
                    print("Warning: EVENT/gen_mass_logits already exists, overwriting...")
                    del f_out['EVENT/gen_mass_logits']
                
                f_out.create_dataset(
                    'EVENT/gen_mass_logits', 
                    data=gen_mass_logits, 
                    compression='gzip',
                    shuffle=True
                )
                print(f"Added EVENT/gen_mass_logits to {temp_path}")
                print(f"  Shape: {gen_mass_logits.shape}")
                print(f"  Dtype: {gen_mass_logits.dtype}")
                print(f"  Sample logits (first 5 samples):")
                for i in range(min(5, len(gen_mass_logits))):
                    class_idx = np.argmax(gen_mass_logits[i])
                    print(f"    Sample {i}: mass={mass_data[i]:.2f}, class={mass_classes[class_idx]}, logit_idx={class_idx}")
        
        # If inplace, replace original file
        if inplace:
            temp_path.replace(h5_path)
            print(f"Updated {h5_path} in place")
        else:
            print(f"Saved output to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {h5_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Clean up temp file if it exists
        if inplace and temp_path.exists():
            temp_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Add gen_mass_logits to h5 files based on mass categories'
    )
    parser.add_argument(
        'h5_file',
        type=str,
        help='Path to the input h5 file'
    )
    parser.add_argument(
        '--mass-classes',
        type=str,
        default='200,300,400,500,600,700,800,900,1000',
        help='Comma-separated list of mass classes (default: 200,300,400,500,600,700,800,900,1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (if not specified, modifies file in place)'
    )
    parser.add_argument(
        '--no-inplace',
        action='store_true',
        help='Do not modify file in place (requires --output)'
    )
    
    args = parser.parse_args()
    
    # Parse mass classes
    mass_classes = [int(x.strip()) for x in args.mass_classes.split(',')]
    
    # Determine if inplace
    inplace = not args.no_inplace and args.output is None
    
    success = add_gen_mass_logits(
        args.h5_file,
        mass_classes=mass_classes,
        output_path=args.output,
        inplace=inplace
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
