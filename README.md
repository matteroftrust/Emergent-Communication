# Emergent-Communication


1. Make sure `Python 3` and `pip` is installed. Install `virtualenv`.
```
pip install virtualenv
```

1. Create virtual environment and activate.

```
virtualenv env
source env/bin/activate
```

2. Clone git repository.

```
git clone https://github.com/matteroftrust/Emergent-Communication.git
```

3. Import all required packages using `pip install`.

```
pip install -r requirements.txt
```

If `tensorflow-gpu` cannot be installed try `tensorflow`.

4. Run simulation

```
python main.py    [-e] number of episodes
                  [-b] batch size
                  [-t] test batch size
                  [-c] communication channels: proposal, linguistic (comma separated format)
```

5. Simulation results should be saved in `\results` folder. To run data processing in project directory run:

```
python analyze.py [-f] .pkl filename saved in \results.
```

Plots should be saved in `figs` folder
