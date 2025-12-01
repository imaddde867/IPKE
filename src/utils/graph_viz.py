import json
import sys
from pathlib import Path

def print_graph_structure(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Invalid JSON file.")
        return

    # Create a lookup for step text
    steps_lookup = {s['id']: s['text'] for s in data.get('steps', [])}
    constraints = data.get('constraints', [])
    entities = data.get('entities', [])

    print(f"\n=== KNOWLEDGE GRAPH: {path.stem} ===")
    print(f"Stats: {len(steps_lookup)} Steps | {len(constraints)} Constraints | {len(entities)} Entities")
    
    print("\n--- PROCEDURAL FLOW (Step -> Next Step) ---")
    # Sort steps by order if available, else by ID
    sorted_steps = sorted(data.get('steps', []), key=lambda x: x.get('order', 999))
    
    for i in range(len(sorted_steps)-1):
        curr = sorted_steps[i]
        nxt = sorted_steps[i+1]
        print(f"  [Step {curr.get('id')}] --> [Step {nxt.get('id')}]")
        # Print first 60 chars of step text
        print(f"     \"{curr.get('text', '')[:60]}...\"")

    print("\n--- LOGIC LAYER (Constraint -> Step) ---")
    linked_count = 0
    for c in constraints:
        c_id = c.get('id', '??')
        c_text = c.get('expression', c.get('text', 'Unknown'))
        # Clean text for display
        c_text = c_text.replace('\n', ' ')[:50]
        
        # CHECK BOTH KEYS: 'attached_to' (raw) and 'steps' (normalized)
        attached = c.get('attached_to') or c.get('steps') or []
        
        if isinstance(attached, str): attached = [attached]
        
        if not attached:
            print(f"  [Constraint {c_id}] (Unlinked): \"{c_text}...\"")
        else:
            linked_count += 1
            for target in attached:
                # Verify the target step exists
                if target in steps_lookup:
                    print(f"  [Constraint {c_id}] --guards--> [Step {target}]")
                    print(f"     Rule: \"{c_text}...\"")
                else:
                    print(f"  [Constraint {c_id}] --(broken link)--> {target}")

    print(f"\nTotal Linked Constraints: {linked_count} / {len(constraints)}")

if __name__ == "__main__":
    # Default to the 3M document from P3
    base_path = "logs/prompting_grid/P3_two_stage/3M_OEM_SOP/predictions.json"
    
    # Allow command line override
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
        
    print_graph_structure(base_path)
