from flask import Flask, request, jsonify
from flask_cors import CORS
import pulp

app = Flask(__name__)
CORS(app)  # Allows Base44 to talk to Render without browser security blocks

# Accept both 'z' and 's' spellings, and allow 'OPTIONS' test checks
@app.route('/optimize', methods=['POST', 'OPTIONS'])
@app.route('/optimise', methods=['POST', 'OPTIONS'])
def optimize_team():
    # Handle browser security preflight check
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json() or {}
    
    players = data.get('players', []) # List: [{id, name, position, price, xp}, ...]
    budget = data.get('budget', 100.0)
    
    if not players:
        return jsonify({"status": "error", "message": "No players provided"}), 400

    prob = pulp.LpProblem("FPL_Starting_XI_Optimizer", pulp.LpMaximize)
    
    # Decision Variables
    x = {p['id']: pulp.LpVariable(f"x_{p['id']}", cat='Binary') for p in players}
    y = {p['id']: pulp.LpVariable(f"y_{p['id']}", cat='Binary') for p in players}
    
    # Helper function to get expected points safely
    def get_xp(p):
        val = p.get('xp') if p.get('xp') is not None else p.get('xP', 0.0)
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    # Helper function to get price safely
    def get_price(p):
        val = p.get('price') if p.get('price') is not None else p.get('cost', 0.0)
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    # Helper function to get position safely
    def get_pos(p):
        return p.get('position') or p.get('pos') or ''

    # OBJECTIVE: Maximize total regular points plus double points for the captain
    prob += pulp.lpSum(get_xp(p) * x[p['id']] + get_xp(p) * y[p['id']] for p in players)
    
    # CONSTRAINT: Total Budget
    prob += pulp.lpSum(get_price(p) * x[p['id']] for p in players) <= budget
    
    # CONSTRAINT: Total starting lineup size = 11 players
    prob += pulp.lpSum(x[p['id']] for p in players) == 11
    
    # CONSTRAINT: Exactly 1 Goalkeeper
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'GKP') == 1
    
    # CONSTRAINTS: Outfield position limits
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'DEF') <= 5
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'MID') <= 5
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'FWD') >= 1
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'FWD') <= 3
    
    # Ensure total outfielders (DEF + MID + FWD) equals exactly 10
    prob += (
        pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'DEF') +
        pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'MID') +
        pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'FWD')
    ) == 10

    # CONSTRAINT: Exactly ONE captain across the starting XI
    prob += pulp.lpSum(y[p['id']] for p in players) == 1
    
    # CONSTRAINT: A player can only be captain if they are in the starting XI
    for p in players:
        prob += y[p['id']] <= x[p['id']]
        
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract results
    starting_xi = [p for p in players if x[p['id']].varValue and x[p['id']].varValue > 0.5]
    captain = next((p for p in players if y[p['id']].varValue and y[p['id']].varValue > 0.5), None)
    
    selected_ids = [p['id'] for p in starting_xi]
    captain_id = captain['id'] if captain else None

    total_cost = sum(get_price(p) for p in starting_xi)
    total_xp = sum(get_xp(p) for p in starting_xi) + (get_xp(captain) if captain else 0.0)

    return jsonify({
        "status": "success",
        "selected_player_ids": selected_ids,
        "captain_id": captain_id,
        "starting_xi": starting_xi,
        "captain": captain,
        "total_cost": round(total_cost, 1),
        "total_xp": round(total_xp, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
