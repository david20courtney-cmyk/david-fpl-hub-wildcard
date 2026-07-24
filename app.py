from flask import Flask, request, jsonify
from flask_cors import CORS
import pulp

app = Flask(__name__)
CORS(app)

@app.route('/optimize', methods=['POST', 'OPTIONS'])
@app.route('/optimise', methods=['POST', 'OPTIONS'])
def optimize_team():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json() or {}
    
    players = data.get('players', [])
    budget = float(data.get('budget', 100.0))
    
    if not players:
        return jsonify({"status": "error", "message": "No players provided"}), 400

    prob = pulp.LpProblem("FPL_Starting_XI_Optimizer", pulp.LpMaximize)
    
    x = {p['id']: pulp.LpVariable(f"x_{p['id']}", cat='Binary') for p in players}
    y = {p['id']: pulp.LpVariable(f"y_{p['id']}", cat='Binary') for p in players}
    
    # SAFE VALUE EXTRACTION
    def get_xp(p):
        val = p.get('xp') if p.get('xp') is not None else p.get('xP', 0.0)
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def get_price(p):
        val = p.get('price') if p.get('price') is not None else p.get('cost', p.get('now_cost', 0.0))
        try:
            val = float(val)
            # Handle standard FPL prices (e.g. 100 = 10.0m or 85 = 8.5m)
            return val / 10.0 if val > 20.0 else val
        except (ValueError, TypeError):
            return 0.0

    def get_pos(p):
        pos = p.get('position') or p.get('pos') or p.get('element_type') or ''
        pos_str = str(pos).strip().upper()
        
        # Handle FPL raw integer IDs (1=GKP, 2=DEF, 3=MID, 4=FWD)
        if pos_str in ['1', 'GK', 'GKP', 'GOALKEEPER']:
            return 'GKP'
        elif pos_str in ['2', 'DEF', 'DEFENDER']:
            return 'DEF'
        elif pos_str in ['3', 'MID', 'MIDFIELDER']:
            return 'MID'
        elif pos_str in ['4', 'FWD', 'FORWARD', 'ATTACKER']:
            return 'FWD'
        return pos_str

    def get_team(p):
        return str(p.get('team') or p.get('team_name') or p.get('club') or '').strip()

    # OBJECTIVE
    prob += pulp.lpSum(get_xp(p) * x[p['id']] + get_xp(p) * y[p['id']] for p in players)
    
    # CONSTRAINTS
    # 1. Budget
    prob += pulp.lpSum(get_price(p) * x[p['id']] for p in players) <= budget
    
    # 2. Total starting XI size = 11
    prob += pulp.lpSum(x[p['id']] for p in players) == 11
    
    # 3. Positional limits
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'GKP') == 1
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'DEF') >= 3
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'DEF') <= 5
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'MID') >= 2
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'MID') <= 5
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'FWD') >= 1
    prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'FWD') <= 3

    # 4. Max 3 players per team (only enforced if team data is present)
    teams = set(get_team(p) for p in players if get_team(p))
    for team in teams:
        prob += pulp.lpSum(x[p['id']] for p in players if get_team(p) == team) <= 3

    # 5. Captain constraints
    prob += pulp.lpSum(y[p['id']] for p in players) == 1
    for p in players:
        prob += y[p['id']] <= x[p['id']]
        
    # SOLVE
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Check if PuLP found a solution
    if status != 1:  # 1 = Optimal solution found
        return jsonify({"status": "error", "message": "Infeasible model"}), 400

    # EXTRACT RESULTS
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
