import numpy as np

import emergent


if __name__ == '__main__':

    term_response = False
    prop_response = np.array([0, 0, 0])
    utter_response = np.array([0] * 11)

    batch_size = 128
    test_batch_size = 5
    episode_num = 2000
    test_every = 20

    item_num = 3
    prosocial = False
    save_as = 'static_test'

    agent_settings = emergent.settings.AgentSettings()

    print('Agents initialization')

    agents = [emergent.agents.Agent(**agent_settings.as_dict()),
              emergent.test.TestStaticAgent(term_response=term_response, prop_response=prop_response,
                                            utter_response=utter_response, n=2, hidden_state_size=100)]
    print('Game initialization')
    game = emergent.game.Game(agents=agents, batch_size=batch_size, test_batch_size=test_batch_size, episode_num=episode_num,
                              item_num=item_num, prosocial=prosocial, test_every=test_every)
    game.play(save_as=save_as)
