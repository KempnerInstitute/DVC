"""
Fix vine_model.py to Pass tr Parameter

This script fixes vine_model.py to pass the 'tr' parameter to evaluate_fit
so that the kernel_cdf transformation is properly applied.
"""

import re


def fix_vine_model():
    """Fix vine_model.py to pass tr parameter to evaluate_fit"""
    
    # Read the current vine_model.py
    with open('src/DVC/vine_model.py', 'r') as f:
        content = f.read()
    
    # Find and fix the first evaluate_fit call (around line 576)
    # Original:
    # pd_grid, cdf_grid, _, gu, gv = evaluate_fit(
    #     {"data_s": sub_s, "data_x": sub_x},
    #     {"grid_u": vine.grid_u, "grid_s": vine.grid_s, "grid_x": grid_x_sub},
    #     {"bw": bw_fin, "n_cop": subE, "batch": opt_cfg["batch_size"], "grad_precompute": npc_cfg.get("grad_precompute", False)})
    
    # Add tr, theta, and theta_flip to the call
    pattern1 = r'(pd_grid, cdf_grid, _, gu, gv = evaluate_fit\(\s*' \
               r'\{"data_s": sub_s, "data_x": sub_x\},\s*' \
               r'\{"grid_u": vine\.grid_u, "grid_s": vine\.grid_s, "grid_x": grid_x_sub\},\s*' \
               r'\{"bw": bw_fin, "n_cop": subE, "batch": opt_cfg\["batch_size"\], "grad_precompute": npc_cfg\.get\("grad_precompute", False\)\}\))'
    
    replacement1 = 'pd_grid, cdf_grid, theta_ret, gu, gv = evaluate_fit(\n' \
                   '                        {"data_s": sub_s, "data_x": sub_x, "theta": vine.theta, "theta_flip": vine.theta_flip},\n' \
                   '                        {"grid_u": vine.grid_u, "grid_s": vine.grid_s, "grid_x": grid_x_sub},\n' \
                   '                        {"bw": bw_fin, "n_cop": subE, "batch": opt_cfg["batch_size"], "tr": tr, "grad_precompute": npc_cfg.get("grad_precompute", False)})'
    
    content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE | re.DOTALL)
    
    # Find and fix the second evaluate_fit call (around line 662)
    pattern2 = r'(pd_grid, cdf_grid, _, gu, gv = evaluate_fit\(\s*' \
               r"\{'data_s': pair_data_s, 'data_x': pair_data_x\},\s*" \
               r"\{'grid_u': vine\.grid_u, 'grid_s': vine\.grid_s, 'grid_x': grid_x\[:,:,0:1\]\},\s*" \
               r"\{'bw': bw_final, 'n_cop': 1, 'batch': 5, 'grad_precompute': npc_cfg\.get\('grad_precompute', False\)\}\s*\))"
    
    replacement2 = "pd_grid, cdf_grid, theta_ret, gu, gv = evaluate_fit(\n" \
                   "                            {'data_s': pair_data_s, 'data_x': pair_data_x, 'theta': vine.theta, 'theta_flip': vine.theta_flip},\n" \
                   "                            {'grid_u': vine.grid_u, 'grid_s': vine.grid_s, 'grid_x': grid_x[:,:,0:1]},\n" \
                   "                            {'bw': bw_final, 'n_cop': 1, 'batch': 5, 'tr': tr, 'ind_edge_rel': [j], 'grad_precompute': npc_cfg.get('grad_precompute', False)}\n" \
                   "                        )"
    
    content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE | re.DOTALL)
    
    # Save the fixed version
    with open('src/DVC/vine_model.py.fixed', 'w') as f:
        f.write(content)
    
    print("Fixed vine_model.py saved to: src/DVC/vine_model.py.fixed")
    print("\nTo apply:")
    print("cp src/DVC/vine_model.py src/DVC/vine_model.py.backup")
    print("cp src/DVC/vine_model.py.fixed src/DVC/vine_model.py")


def create_simple_fix():
    """Create a simpler fix by modifying specific lines"""
    
    print("\n=== CREATING SIMPLE LINE-BY-LINE FIX ===")
    
    with open('src/DVC/vine_model.py', 'r') as f:
        lines = f.readlines()
    
    # Fix line 576 area - first evaluate_fit call
    for i in range(len(lines)):
        if 'pd_grid, cdf_grid, _, gu, gv = evaluate_fit(' in lines[i]:
            # Find the complete call
            j = i
            while ')' not in lines[j] or '{' in lines[j]:
                j += 1
            
            # Check if this is the first call (has sub_s)
            call_text = ''.join(lines[i:j+1])
            if 'sub_s' in call_text and '"tr"' not in call_text:
                print(f"Found first evaluate_fit call at line {i+1}")
                
                # Modify the data_dict line
                for k in range(i, j+1):
                    if '"data_s": sub_s' in lines[k]:
                        lines[k] = lines[k].replace(
                            '{"data_s": sub_s, "data_x": sub_x}',
                            '{"data_s": sub_s, "data_x": sub_x, "theta": vine.theta, "theta_flip": vine.theta_flip}'
                        )
                    elif '"bw": bw_fin' in lines[k]:
                        lines[k] = lines[k].replace(
                            '"grad_precompute": npc_cfg.get("grad_precompute", False)})',
                            '"tr": tr, "grad_precompute": npc_cfg.get("grad_precompute", False)})'
                        )
                
                # Change the assignment
                lines[i] = lines[i].replace('_, gu, gv', 'theta_ret, gu, gv')
                break
    
    # Fix line 662 area - second evaluate_fit call
    for i in range(len(lines)):
        if 'pd_grid, cdf_grid, _, gu, gv = evaluate_fit(' in lines[i]:
            # Find the complete call
            j = i
            while ')' not in lines[j] or '{' in lines[j]:
                j += 1
            
            # Check if this is the second call (has pair_data_s)
            call_text = ''.join(lines[i:j+1])
            if 'pair_data_s' in call_text and '"tr"' not in call_text:
                print(f"Found second evaluate_fit call at line {i+1}")
                
                # Modify the lines
                for k in range(i, j+1):
                    if "'data_s': pair_data_s" in lines[k]:
                        lines[k] = lines[k].replace(
                            "{'data_s': pair_data_s, 'data_x': pair_data_x}",
                            "{'data_s': pair_data_s, 'data_x': pair_data_x, 'theta': vine.theta, 'theta_flip': vine.theta_flip}"
                        )
                    elif "'bw': bw_final" in lines[k]:
                        lines[k] = lines[k].replace(
                            "'grad_precompute': npc_cfg.get('grad_precompute', False)}",
                            "'tr': tr, 'ind_edge_rel': [j], 'grad_precompute': npc_cfg.get('grad_precompute', False)}"
                        )
                
                # Change the assignment
                lines[i] = lines[i].replace('_, gu, gv', 'theta_ret, gu, gv')
                break
    
    # Save the fixed version
    with open('src/DVC/vine_model.py.fixed2', 'w') as f:
        f.writelines(lines)
    
    print("\nSimple fix saved to: src/DVC/vine_model.py.fixed2")


def main():
    """Apply fixes to vine_model.py"""
    print("="*70)
    print("FIXING VINE_MODEL.PY TO PASS TR PARAMETER")
    print("="*70)
    
    # Try the regex fix
    try:
        fix_vine_model()
    except Exception as e:
        print(f"Regex fix failed: {e}")
    
    # Also create simple line-by-line fix
    create_simple_fix()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    print("\n1. Back up original:")
    print("   cp src/DVC/vine_model.py src/DVC/vine_model.py.backup")
    
    print("\n2. Apply the fix:")
    print("   cp src/DVC/vine_model.py.fixed2 src/DVC/vine_model.py")
    
    print("\n3. Test the complete fix")


if __name__ == "__main__":
    main() 