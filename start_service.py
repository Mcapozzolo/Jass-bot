import os
from jass.service.player_service_app import PlayerServiceApp
from my_agent import MyAgent
from my_agentcomplex import MyAgentcomplex

app = PlayerServiceApp(__name__)
app.add_player('GruppeMarcoPatrik', MyAgentcomplex())

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
