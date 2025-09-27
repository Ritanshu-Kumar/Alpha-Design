# python alphadesign.py --config config.json --base-params base_params.json
import argparse
import json
import sys
from dataclasses import asdict
from formula_constraints import F1FrontWingParams
from main_pipeline import AlphaDesignPipeline
import os

def load_base_parameters(params_path: str) -> F1FrontWingParams:
    
    if params_path and os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params_dict = json.load(f)
            return F1FrontWingParams(**params_dict)
    
    return F1FrontWingParams(
        total_span=1600, root_chord=280, tip_chord=250, chord_taper_ratio=0.89,
        sweep_angle=3.5, dihedral_angle=2.5, twist_distribution_range=[-1.5, 0.5],
        base_profile="NACA_64A010_modified", max_thickness_ratio=0.15,
        camber_ratio=0.08, camber_position=0.40, leading_edge_radius=2.8,
        trailing_edge_thickness=2.5, upper_surface_radius=800, lower_surface_radius=1100,
        flap_count=3, flap_spans=[1600, 1500, 1400], flap_root_chords=[220, 180, 140],
        flap_tip_chords=[200, 160, 120], flap_cambers=[0.12, 0.10, 0.08],
        flap_slot_gaps=[14, 12, 10], flap_vertical_offsets=[25, 45, 70],
        flap_horizontal_offsets=[30, 60, 85], endplate_height=280,
        endplate_max_width=120, endplate_min_width=40, endplate_thickness_base=10,
        endplate_forward_lean=6, endplate_rearward_sweep=10, endplate_outboard_wrap=18,
        footplate_extension=70, footplate_height=30, arch_radius=130,
        footplate_thickness=5, primary_strake_count=2, strake_heights=[45, 35],
        y250_width=500, y250_step_height=18, y250_transition_length=80,
        central_slot_width=30, pylon_count=2, pylon_spacing=320,
        pylon_major_axis=38, pylon_minor_axis=25, pylon_length=120,
        cascade_enabled=True, primary_cascade_span=250, primary_cascade_chord=55,
        secondary_cascade_span=160, secondary_cascade_chord=40,
        wall_thickness_structural=4, wall_thickness_aerodynamic=2.5,
        wall_thickness_details=2.0, minimum_radius=0.4, mesh_resolution_aero=0.4,
        mesh_resolution_structural=0.6, resolution_span=40, resolution_chord=25,
        mesh_density=1.5, surface_smoothing=True, material="Standard Carbon Fiber",
        density=1600, weight_estimate=4.0, target_downforce=4000, target_drag=40,
        efficiency_factor=1.0   
    )

def main():
    parser = argparse.ArgumentParser(description='AlphaDesign F1 Wing Optimization')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--base-params', help='Base parameters JSON file')
    parser.add_argument('--resume-from', help='Resume from checkpoint directory')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup without running')
    
    args = parser.parse_args()
    
    try:
        base_params = load_base_parameters(args.base_params)
        print(f"✅ Base parameters loaded")
        
        pipeline = AlphaDesignPipeline(args.config)
        
        if args.dry_run:
            print("🧪 Dry run completed successfully")
            return
        
        print("🚀 Starting AlphaDesign optimization...")
        results = pipeline.run_complete_pipeline(base_params)
        
        print("\n" + "="*60)
        print("🏆 ALPHADESIGN OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"📊 Total Generations: {results.get('total_generations', 0)}")
        print(f"🧬 Designs Generated: {results['summary']['total_designs_generated']}")
        print(f"⏰ Early Stopped: {results['summary']['early_stopped']}")
        print(f"📁 Output Directory: {pipeline.output_dirs['stl_outputs']}")
        print(f"💾 Best Designs: {len(results['summary']['best_designs_stl'])} STL files")
        
        return 0
        
    except Exception as e:
        print(f"💥 Pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
