"""Multi-turn state retention and session separation with a mock chatbot."""
class DemoChatbot:
    def __init__(self):
        self.sessions = {}

    def reply(self, session_id: str, message: str) -> str:
        if message.startswith('My name is '):
            self.sessions[session_id] = message.removeprefix('My name is ').rstrip('.')
            return 'Nice to meet you.'
        if message == 'What is my name?':
            return self.sessions.get(session_id, "I don't know your name.")
        return 'Please clarify.'


def test_multiturn_context_retention():
    bot = DemoChatbot()
    bot.reply('a', 'My name is Alex.')
    assert bot.reply('a', 'What is my name?') == 'Alex'


def test_session_isolation():
    bot = DemoChatbot()
    bot.reply('a', 'My name is Alex.')
    assert bot.reply('b', 'What is my name?') == "I don't know your name."
