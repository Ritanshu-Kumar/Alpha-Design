import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
from formula_constraints import F1FrontWingAnalyzer, F1FrontWingParams
from cfd_analysis import STLWingAnalyzer
from wing_generator import UltraRealisticF1FrontWingGenerator
import os
import tempfile
import signal

class FitnessEval:
    def __init__(self, weight_constraints=0.3, weight_perf=0.4, weight_cfd=0.3):
        self.weight_constraints = weight_constraints
        self.weight_perf = weight_perf
        self.weight_cfd = weight_cfd
        self.temp_dir = tempfile.mkdtemp()
        
        # Enhanced fitness evaluation parameters
        self.advanced_validation_weights = {
            'safety_factor_critical': 0.15,      # Critical safety requirements
            'structural_dynamics': 0.10,         # Natural frequency, buckling
            'aerodynamic_quality': 0.12,         # Reynolds effects, flow quality
            'manufacturing_feasibility': 0.08,   # Realistic production constraints
            'regulatory_compliance': 0.20,       # FIA regulations
            'performance_targets': 0.25,         # Downforce/drag/efficiency targets
            'multi_element_effectiveness': 0.10   # Advanced multi-element physics
        }

    def should_skip_cfd(self, constraint_score):
        # Enhanced CFD skipping logic with more criteria
        basic_compliance = constraint_score['constraint_compliance'] < 0.4
        
        # Skip if critical safety factors are too low
        safety_critical = constraint_score.get('safety_factor', 0) < 1.5
        
        # Skip if structural analysis shows major issues
        structural_critical = (
            constraint_score.get('buckling_safety_factor', float('inf')) < 1.5 or
            constraint_score.get('natural_frequency', 0) < 15
        )
        
        return basic_compliance or safety_critical or structural_critical

    def evaluate_pop_with_progress(self, population, pbar: tqdm):
        fitness_scores = []
        cfd_skipped_count = 0
        advanced_metrics = {
            'critical_failures': 0,
            'safety_warnings': 0,
            'performance_excellent': 0
        }

        for i, individual in enumerate(population):
            pbar.set_description(f"📊 Evaluating Individual {i + 1}")
            
            try:
                constraint_score = self.evaluate_formula_constratins(individual)
                
                # Enhanced skip criteria with detailed tracking
                if self.should_skip_cfd(constraint_score):
                    cfd_skipped_count += 1
                    advanced_metrics['critical_failures'] += 1
                    pbar.set_postfix({'Status': 'Critical Failure - CFD Skipped'})
                    
                    # Enhanced penalty scores with structural considerations
                    fitness = {
                        'total_fitness': constraint_score['constraint_compliance'] * 25,  # Reduced penalty
                        'constraint_fitness': constraint_score['constraint_compliance'] * 100,
                        'performance_fitness': 0,
                        'cfd_fitness': 0,
                        'constraint_compliance': constraint_score['constraint_compliance'],
                        'computed_downforce': constraint_score.get('computed_downforce', 0),
                        'computed_efficiency': constraint_score.get('computed_efficiency', 0),
                        'safety_factor': constraint_score.get('safety_factor', 0),
                        'structural_score': constraint_score.get('structural_score', 0),
                        'aerodynamic_score': constraint_score.get('aerodynamic_score', 0),
                        'cfd_downforce': 0,
                        'cfd_efficiency': 0,
                        'valid': False,
                        'cfd_skipped': True,
                        'failure_reason': constraint_score.get('failure_reason', 'Multiple constraints failed')
                    }
                elif constraint_score['constraint_compliance'] > 0.75:
                    # High-quality designs get full CFD analysis
                    pbar.set_postfix({'Status': 'Premium CFD Analysis'})
                    cfd_score = self.evaluate_cfd_perf(individual, i)
                    fitness = self.combine_scores(constraint_score, cfd_score)
                    fitness['cfd_skipped'] = False
                    
                    if fitness['total_fitness'] > 80:
                        advanced_metrics['performance_excellent'] += 1
                        
                elif constraint_score['constraint_compliance'] > 0.6:
                    # Standard CFD analysis
                    pbar.set_postfix({'Status': 'Standard CFD Analysis'})
                    cfd_score = self.evaluate_cfd_perf(individual, i)
                    fitness = self.combine_scores(constraint_score, cfd_score)
                    fitness['cfd_skipped'] = False
                else:
                    # Marginal designs - skip CFD but track warnings
                    pbar.set_postfix({'Status': 'Constraints Marginal'})
                    advanced_metrics['safety_warnings'] += 1
                    fitness = self.combine_scores(constraint_score, {'cfd_valid': False})
                    fitness['cfd_skipped'] = True

                fitness_scores.append(fitness)
                
                if isinstance(fitness, dict):
                    pbar.set_postfix({
                        'Status': 'Complete', 
                        'Fitness': f"{fitness.get('total_fitness', 0):.1f}",
                        'Safety': f"{fitness.get('safety_factor', 0):.1f}",
                        'Skipped': cfd_skipped_count
                    })
                else:
                    pbar.set_postfix({'Status': 'Complete', 'Fitness': f"{fitness:.1f}"})
                
            except Exception as e:
                pbar.set_postfix({'Status': 'Analysis Error'})
                advanced_metrics['critical_failures'] += 1
                fitness_scores.append({
                    'total_fitness': -1000,
                    'constraint_fitness': 0,
                    'performance_fitness': 0,
                    'cfd_fitness': 0,
                    'constraint_compliance': 0,
                    'safety_factor': 0,
                    'valid': False,
                    'cfd_skipped': True,
                    'error': str(e)
                })
            
            pbar.update(1)

        # Enhanced reporting
        total_pop = len(population)
        cfd_run_count = total_pop - cfd_skipped_count
        print(f"📈 Enhanced CFD Efficiency: {cfd_run_count}/{total_pop} analyses run ({cfd_skipped_count} skipped)")
        print(f"🔥 Critical Failures: {advanced_metrics['critical_failures']}")
        print(f"⚠️  Safety Warnings: {advanced_metrics['safety_warnings']}")
        print(f"⭐ Excellent Designs: {advanced_metrics['performance_excellent']}")

        return fitness_scores

    def evaluate_pop(self, population):
        fitness_scores = []

        for i, individual in enumerate(population):
            print(f"🏎️  Evaluating individual {i + 1} of {len(population)}...")

            try:
                constraint_score = self.evaluate_formula_constratins(individual)

                if self.should_skip_cfd(constraint_score):
                    print(f"Individual {i + 1} critical constraints failed - skipping CFD")
                    fitness = {
                        'total_fitness': constraint_score['constraint_compliance'] * 25,
                        'constraint_fitness': constraint_score['constraint_compliance'] * 100,
                        'performance_fitness': 0,
                        'cfd_fitness': 0,
                        'valid': False,
                        'cfd_skipped': True,
                        'safety_factor': constraint_score.get('safety_factor', 0)
                    }
                elif constraint_score['constraint_compliance'] > 0.6:
                    cfd_score = self.evaluate_cfd_perf(individual, i)
                    fitness = self.combine_scores(constraint_score, cfd_score)
                    fitness['cfd_skipped'] = False
                else:
                    print(f"Individual {i + 1} marginal constraints - skipping CFD")
                    fitness = self.combine_scores(constraint_score, {'cfd_valid': False})
                    fitness['cfd_skipped'] = True

                fitness_scores.append(fitness)
                
            except Exception as e:
                print(f"❌ Error evaluating individual {i}: {e}")
                fitness_scores.append({
                    'total_fitness': -1000,
                    'constraint_fitness': 0,
                    'performance_fitness': 0,
                    'cfd_fitness': 0,
                    'valid': False,
                    'cfd_skipped': True,
                    'error': str(e)
                })

        return fitness_scores
    
    def evaluate_formula_constratins(self, individual):
        try:
            params = F1FrontWingParams(**individual)
            analyzer = F1FrontWingAnalyzer(params)
            results = analyzer.run_complete_analysis()

            # Enhanced validation with detailed constraint checking
            validations = results.get('validation_results', {})
            computed_vals = results.get('computed_values', {})

            # Advanced compliance bonuses based on specific validations
            compliance_bonus = 0
            
            # Aerodynamic performance bonuses
            if validations.get('flap_gap_optimal', False):
                compliance_bonus += 0.08
            if validations.get('flap_attachment', False):
                compliance_bonus += 0.08
            if validations.get('multi_element_effective', False):
                compliance_bonus += 0.06
            if validations.get('ground_effect_beneficial', False):
                compliance_bonus += 0.05
            
            # Performance target bonuses
            if validations.get('downforce_target_met', False):
                compliance_bonus += 0.12
            if validations.get('drag_target_met', False):
                compliance_bonus += 0.08
            if validations.get('efficiency_target_met', False):
                compliance_bonus += 0.10
            
            # Critical safety bonuses
            if validations.get('safety_factor_adequate', False):
                compliance_bonus += 0.10
            if validations.get('buckling_safe', False):
                compliance_bonus += 0.06
            if validations.get('fatigue_safety_adequate', False):
                compliance_bonus += 0.06
            
            # Regulatory compliance bonuses
            if validations.get('span_regulation_compliant', False):
                compliance_bonus += 0.05
            if validations.get('y250_compliance', False):
                compliance_bonus += 0.05
            
            # Manufacturing feasibility bonuses
            if validations.get('wall_thickness_feasible', False):
                compliance_bonus += 0.04
            if validations.get('minimum_radius_valid', False):
                compliance_bonus += 0.03

            adjusted_compliance = min(1.0, results.get('overall_compliance', 0) + compliance_bonus)

            # Enhanced scoring with advanced metrics
            structural_score = self._compute_structural_score(validations, computed_vals)
            aerodynamic_score = self._compute_aerodynamic_score(validations, computed_vals)
            
            # Failure reason tracking
            failure_reasons = []
            if computed_vals.get('safety_factor', 0) < 2.0:
                failure_reasons.append('Insufficient safety factor')
            if computed_vals.get('natural_frequency', 0) < 25:
                failure_reasons.append('Low natural frequency')
            if computed_vals.get('total_downforce', 0) < 500:
                failure_reasons.append('Inadequate downforce')

            return {
                'constraint_compliance': adjusted_compliance,
                'constraint_percentage': results.get('compliance_percentage', 0),
                'computed_downforce': computed_vals.get('total_downforce', 0),
                'computed_drag': computed_vals.get('total_drag', 0),
                'computed_efficiency': computed_vals.get('efficiency_computed', 0),
                'safety_factor': computed_vals.get('safety_factor', 0),
                'buckling_safety_factor': computed_vals.get('buckling_safety_factor', float('inf')),
                'natural_frequency': computed_vals.get('natural_frequency', 0),
                'total_mass': computed_vals.get('total_computed_mass', 0),
                'ground_effect_factor': computed_vals.get('total_ground_effect', 1.0),
                'reynolds_main': computed_vals.get('reynolds_main', 0),
                'structural_score': structural_score,
                'aerodynamic_score': aerodynamic_score,
                'analysis_quality': results.get('analysis_quality', 'Unknown'),
                'constraint_valid': True,
                'failure_reason': '; '.join(failure_reasons) if failure_reasons else None,
                'recommendations': results.get('recommendations', [])
            }
            
        except Exception as e:
            print(f"❌ Error in enhanced constraint evaluation: {e}")
            return {
                'constraint_compliance': 0,
                'constraint_percentage': 0,
                'computed_downforce': 0,
                'computed_drag': 1000,
                'computed_efficiency': 0,
                'safety_factor': 0,
                'buckling_safety_factor': 0,
                'natural_frequency': 0,
                'total_mass': 10,  # High penalty mass
                'structural_score': 0,
                'aerodynamic_score': 0,
                'constraint_valid': False,
                'failure_reason': f'Analysis failed: {str(e)}',
                'error': str(e)
            }

    def _compute_structural_score(self, validations, computed_vals):
        """Compute advanced structural quality score"""
        score = 0
        max_score = 100
        
        # Safety factor contribution (40% of structural score)
        safety_factor = computed_vals.get('safety_factor', 0)
        if safety_factor >= 3.0:
            score += 40
        elif safety_factor >= 2.5:
            score += 35
        elif safety_factor >= 2.0:
            score += 25
        elif safety_factor >= 1.5:
            score += 10
        
        # Dynamic response contribution (30% of structural score)
        nat_freq = computed_vals.get('natural_frequency', 0)
        if nat_freq >= 50:
            score += 30
        elif nat_freq >= 35:
            score += 25
        elif nat_freq >= 25:
            score += 15
        elif nat_freq >= 15:
            score += 5
        
        # Buckling safety contribution (20% of structural score)
        buckling_sf = computed_vals.get('buckling_safety_factor', 0)
        if buckling_sf >= 3.0:
            score += 20
        elif buckling_sf >= 2.0:
            score += 15
        elif buckling_sf >= 1.5:
            score += 8
        
        # Weight efficiency contribution (10% of structural score)
        mass = computed_vals.get('total_computed_mass', 10)
        if 2.0 <= mass <= 5.0:  # Optimal F1 wing mass range
            score += 10
        elif 1.5 <= mass <= 6.0:
            score += 7
        elif mass <= 8.0:
            score += 3
        
        return min(score, max_score)
    
    def _compute_aerodynamic_score(self, validations, computed_vals):
        """Compute advanced aerodynamic quality score"""
        score = 0
        max_score = 100
        
        # Efficiency contribution (35% of aero score)
        efficiency = computed_vals.get('efficiency_computed', 0)
        if efficiency >= 8:
            score += 35
        elif efficiency >= 6:
            score += 30
        elif efficiency >= 4:
            score += 20
        elif efficiency >= 3:
            score += 10
        
        # Reynolds number quality (25% of aero score)
        reynolds = computed_vals.get('reynolds_main', 0)
        if reynolds >= 2e6:
            score += 25
        elif reynolds >= 1e6:
            score += 20
        elif reynolds >= 5e5:
            score += 10
        
        # Ground effect utilization (25% of aero score)
        ground_effect = computed_vals.get('total_ground_effect', 1.0)
        if ground_effect >= 2.0:
            score += 25
        elif ground_effect >= 1.5:
            score += 20
        elif ground_effect >= 1.3:
            score += 15
        elif ground_effect >= 1.1:
            score += 8
        
        # Multi-element effectiveness (15% of aero score)
        multi_elem_eff = computed_vals.get('overall_effectiveness', 1.0)
        if multi_elem_eff >= 1.3:
            score += 15
        elif multi_elem_eff >= 1.2:
            score += 12
        elif multi_elem_eff >= 1.1:
            score += 8
        elif multi_elem_eff >= 1.0:
            score += 3
        
        return min(score, max_score)

    def evaluate_cfd_perf(self, individual: Dict, individual_idx: int):
        try:
            from cfd_analysis import STLWingAnalyzer
            
            cfd_dir = "cfd_temp_files"
            os.makedirs(cfd_dir, exist_ok=True)
            
            stl_filename = f"individual_{individual_idx}_wing.stl"
            stl_path = os.path.join(cfd_dir, stl_filename)
            
            wing_generator = UltraRealisticF1FrontWingGenerator(**individual)
            wing_mesh = wing_generator.generate_complete_wing(stl_filename)
            
            expected_path = os.path.join("f1_wing_output", stl_filename)
            if os.path.exists(expected_path):
                import shutil
                shutil.copy2(expected_path, stl_path)
            else:
                return self.get_default_cfd_score()
            
            if os.path.exists(stl_path):
                analyzer = STLWingAnalyzer(stl_path)
                
                # Corrected speed conversion (200 km/h instead of 330)
                speed_ms = analyzer.convert_speed_to_ms(200)  # 200 km/h test speed
                cfd_result = analyzer.multi_element_analysis(speed_ms, 75, 0)
                
                # Enhanced CFD results with additional metrics
                return {
                    'cfd_downforce': float(cfd_result['total_downforce']),
                    'cfd_drag': float(cfd_result['total_drag']),
                    'cfd_efficiency': float(cfd_result['efficiency_ratio']),
                    'stall_margin': float(cfd_result['f1_specific_metrics']['stall_margin']),
                    'balance_coefficient': float(cfd_result['f1_specific_metrics']['balance_coefficient']),
                    'flow_attachment': cfd_result['flow_characteristics']['flow_attachment'],
                    'cfd_valid': True,
                    'cfd_quality': self._assess_cfd_quality(cfd_result)
                }
            else:
                return self.get_default_cfd_score()
                
        except Exception as e:
            print(f"❌ Enhanced CFD evaluation failed: {str(e)}")
            return self.get_default_cfd_score()
    
    def _assess_cfd_quality(self, cfd_result):
        """Assess CFD result quality"""
        downforce = cfd_result.get('total_downforce', 0)
        efficiency = cfd_result.get('efficiency_ratio', 0)
        stall_margin = cfd_result.get('f1_specific_metrics', {}).get('stall_margin', 0)
        
        if downforce > 1000 and efficiency > 6 and stall_margin > 5:
            return 'Excellent'
        elif downforce > 700 and efficiency > 4 and stall_margin > 3:
            return 'Good'
        elif downforce > 400 and efficiency > 2:
            return 'Acceptable'
        else:
            return 'Poor'

    def get_default_cfd_score(self):
        """Return conservative CFD score when analysis fails"""
        return {
            'cfd_downforce': 1000,  # Conservative estimate
            'cfd_drag': 100,
            'cfd_efficiency': 10.0,
            'stall_margin': 5.0,
            'balance_coefficient': 0.25,
            'flow_attachment': 'Unknown',
            'cfd_valid': False,
            'cfd_quality': 'Failed'
        }
     
    def combine_scores(self, constraint_score, cfd_score):
        # Enhanced constraint fitness with advanced weighting
        constraint_fitness = constraint_score['constraint_compliance'] * 100
        
        # Advanced performance fitness incorporating multiple factors
        computed_downforce = constraint_score.get('computed_downforce', 0)
        computed_efficiency = constraint_score.get('computed_efficiency', 0)
        safety_factor = constraint_score.get('safety_factor', 0)
        structural_score = constraint_score.get('structural_score', 0)
        aerodynamic_score = constraint_score.get('aerodynamic_score', 0)
        
        # Multi-dimensional performance fitness
        performance_fitness = min(100, 
            (computed_downforce / 1200) * 35 +  # Downforce contribution (35%)
            computed_efficiency * 25 +           # Efficiency contribution (25%)
            (safety_factor / 4.0) * 20 +        # Safety contribution (20%)
            (structural_score / 100) * 10 +     # Structural quality (10%)
            (aerodynamic_score / 100) * 10      # Aerodynamic quality (10%)
        )

        # Enhanced CFD fitness with quality assessment
        if cfd_score.get('cfd_valid', False):
            cfd_downforce = cfd_score['cfd_downforce']
            cfd_efficiency = cfd_score['cfd_efficiency']
            stall_margin = cfd_score.get('stall_margin', 0)
            cfd_quality = cfd_score.get('cfd_quality', 'Poor')
            
            # Base CFD fitness
            cfd_fitness = min(100, 
                (cfd_downforce / 1500) * 50 +   # CFD downforce (50%)
                cfd_efficiency * 30 +           # CFD efficiency (30%)
                (stall_margin / 10) * 20        # Stall margin safety (20%)
            )
            
            # Quality bonus
            quality_bonus = {
                'Excellent': 15,
                'Good': 10,
                'Acceptable': 5,
                'Poor': 0,
                'Failed': -10
            }.get(cfd_quality, 0)
            
            cfd_fitness += quality_bonus
            cfd_fitness = max(0, min(100, cfd_fitness))  # Clamp to [0, 100]
        else:
            cfd_fitness = 0

        # Advanced total fitness calculation
        total_fitness = (
            self.weight_constraints * constraint_fitness +
            self.weight_perf * performance_fitness +
            self.weight_cfd * cfd_fitness
        )
        
        # Bonus for exceptional designs
        if (constraint_fitness > 90 and performance_fitness > 80 and 
            cfd_fitness > 75 and safety_factor >= 3.0):
            total_fitness += 10  # Exceptional design bonus

        return {
            'total_fitness': total_fitness,
            'constraint_fitness': constraint_fitness,
            'performance_fitness': performance_fitness,
            'cfd_fitness': cfd_fitness,
            'constraint_compliance': constraint_score['constraint_compliance'],
            'computed_downforce': constraint_score.get('computed_downforce', 0),
            'computed_efficiency': constraint_score.get('computed_efficiency', 0),
            'safety_factor': constraint_score.get('safety_factor', 0),
            'structural_score': constraint_score.get('structural_score', 0),
            'aerodynamic_score': constraint_score.get('aerodynamic_score', 0),
            'total_mass': constraint_score.get('total_mass', 0),
            'cfd_downforce': cfd_score.get('cfd_downforce', 0),
            'cfd_efficiency': cfd_score.get('cfd_efficiency', 0),
            'stall_margin': cfd_score.get('stall_margin', 0),
            'cfd_quality': cfd_score.get('cfd_quality', 'Unknown'),
            'analysis_quality': constraint_score.get('analysis_quality', 'Unknown'),
            'valid': constraint_score.get('constraint_valid', False) and cfd_score.get('cfd_valid', False),
            'recommendations': constraint_score.get('recommendations', [])
        }
