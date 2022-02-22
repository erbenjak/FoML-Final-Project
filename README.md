# FoML-Final-Project
Final Project for the for the Fundamentals of ML course winter term 21/22

---
## Group details:
- Jakob, Erben - jakobus.erben@gmail.com - 3727893
- Ebbo, Kramer - *Email einfügen* - *Mat.-Nr. einfügen*
- Vanessa, Zuber - *Email einfügen*  - *Mat.-Nr. einfügen*

---
## Where to find:
- report: *insert folder-adress*
- agent-code: *insert folder-adress*
- example replays: *insert folder-adress*

---
## Important comands:
- run a game: *python main.py play*
- run game with own agent: *python main.py play --my-agent my_agent*
- run a game with multiple custom agents: *python main.py play --agents my_agent random_agent rule_based_agent peaceful_agent*
- for training consider: Use **--train N** (N specifies, that the first N agents passed by **--agents** should be trained)
- training example: *python main.py play --agents my_agent random_agent rule_based_agent peaceful_agent --train 1* **OR** *python main.py play --my-agent my_agent --train 1*
- **CARE:** Training will automatically stop when the last agent to be trained dies. In order to prevent this use the option: *--continue-without-training*
