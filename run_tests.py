import argparse
import datetime as dt

import numpy as np

import emergent


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', dest='filename')
    parser.add_argument('-p', '--prompt', dest='prompt')
    parser.set_defaults(filename=str(dt.datetime.today()).replace(' ', '').replace(':', '').replace('.', ''), prompt='status')
    args = parser.parse_args()
    filename = args.__dict__['filename']

    term_response = True
    prop_response = np.array([0, 0, 0])
    utter_response = np.array([0] * 11)

    batch_size = 128
    test_batch_size = 5
    episode_num = 1000
    test_every = 20

    item_num = 3
    prosocial = False
    save_as = 'static_test'

    proposal_channel = True
    linguistic_channel = False

    agent_settings = emergent.settings.AgentSettings(proposal_channel=proposal_channel, linguistic_channel=linguistic_channel)

    print('Agents initialization')

    agents = [emergent.agents.Agent(**agent_settings.as_dict()),
              emergent.test.TestStaticAgent(term_response=term_response, prop_response=prop_response,
                                            utter_response=utter_response, hidden_state_size=100)]
    print('Agent {} is a dynamic agent.\nAgent {} is a static agent'.format(agents[0].id, agents[1].id))
    print('Game initialization')
    game = emergent.game.Game(agents=agents, batch_size=batch_size, test_batch_size=test_batch_size, episode_num=episode_num,
                              item_num=item_num, prosocial=prosocial, test_every=test_every, filename=filename)
    game.play(save_as=save_as)
