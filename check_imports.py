try:
    import riva.client
    print("riva.client is available")
except ImportError:
    print("riva.client is NOT available")

try:
    import gtts
    print("gtts is available")
except ImportError:
    print("gtts is NOT available")
