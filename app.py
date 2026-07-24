from flask import Flask, request, jsonify
import pulp

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize_team():
    data = request.get_json()
    
    players = data.get('players', []) # List: [{id, name, position, price, xp}, ...]
    budget = data.get('budget', 100.0)
    
    prob = pulp.LpProblem("FPL_Starting_XI_Optimizer", pulp.LpMaximize)
    
    # x_i = 1 if player is in the 11-man starting lineup
    x = {p['id']: pulp.LpVariable(f"x_{p['id']}", cat='Binary') for p in players}
    
    # y_i = 1 if player is chosen as CAPTAIN (must be in the starting XI)
    y = {p['id']: pulp.LpVariable(f"y_{p['id']}", cat='Binary') for p in players}
    
    # OBJECTIVE: Maximize total regular points plus double points for the captain
    prob += pulp.lpSum(p['xp'] * x[p['id']] + p['xp'] * y[p['id']] for p in players)
    
    # CONSTRAINT: Total Budget
    prob += pulp.lpSum(p['price'] * x[p['id']] for p in players) <= budget
    
    # CONSTRAINT: Total starting lineup size = 11 players
    prob += pulp.lpSum(x[p['id']] for p in players) == 11
    
    # CONSTRAINT: Exactly 1 Goalkeeper
    prob += pulp.lpSum(x[p['id']] for p in players if p['position'] == 'GKP') == 1
    
    # CONSTRAINTS: Outfield position limits (Min 1 FWD, Max 3 FWD, Max 5 DEF, Max 5 MID)
    # Total outfielders = 10 (since 1 is GKP, and 1 + 10 = 11 total)
    defs = [x[p['id']] for p in players if p['position'] == 'DEF']
    mids = [x[p['id']] for p in players if p['position'] == 'MID']
    fwds = [p['id'] for p in players if p['position'] == 'FWD']
    
    prob += pulp.lpSum(x[p['id']] for p in players if p['position'] == 'DEF') <= 5
    prob += pulp.lpSum(x[p['id']] for p in players if p['position'] == 'MID') <= 5
    prob += pulp.lpSum(x[p['id']] for p in players if p['position'] == 'FWD') >= 1
    prob += pulp.lpSum(x[p['id']] for p in players if p['position'] == 'FWD') <= 3
    
    # Ensure total outfielders (DEF + MID + FWD) equals exactly 10
    prob += (
        pulp.lpSum(x[p['id']] for p in players if p['position'] == 'DEF') +
        pulp.lpSum(x[p['id']] for p in players if p['position'] == 'MID') +
        pulp.lpSum(x[p['id']] for p in players if p['position'] == 'FWD')
    ) == 10

    # CONSTRAINT: Exactly ONE captain across the starting XI
    prob += pulp.lpSum(y[p['id']] for p in players) == 1
    
    # CONSTRAINT: A player can only be captain if they are in the starting XI
    for p in players:
        prob += y[p['id']] <= x[p['id']]
        
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract results
    selected_ids = [p['id'] for p in players if x[p['id']].varValue > 0.5]
    captain_id = next((p['id'] for p in players if y[p['id']].varValue > 0.5), None)
    
    return jsonify({
        "status": "success",
        "selected_player_ids": selected_ids,
        "captain_id": captain_id
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
