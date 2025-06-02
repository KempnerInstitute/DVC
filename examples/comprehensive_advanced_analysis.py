# Add DVC_NF to path  
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up to DVC_NF directory
sys.path.append(parent_dir) 