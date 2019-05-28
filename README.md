# Emergent-Communication

1. Create virtual environment and activate.

```
virtualenv env
source env/bin/activate
```

2. Import all required packages using `pip install`.
```
pip install -r requirements.txt
```

3. Run simulation

```
python main.py  [-p] ['status', 'all', 'none'] (prompts, shows only main ones / all / none)
                [-v] (validation on)
                [--batch_size]
                [--test_batch_size]
                [--episode_num]
```
