
# FILL IN THESE VALUES WITH YOUR OWN KEYS
client_id = ""
client_secret = ""

# Make sure to add this on "Edit Settings" in your Dashboard
redirect_uri = "http://localhost:9000"

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Connect with API Keys created earlier
scope = "user-read-recently-played"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope,
                                               client_id=client_id,
                                              client_secret=client_secret,
                                              redirect_uri=redirect_uri))

# Running this should open a new tab, click "agree"
results = sp.current_user_recently_played()

type(results)

results.keys()

dict_keys(['items', 'next', 'cursors', 'limit', 'href'])
for idx, item in enumerate(results['items']):
    track_id = item['track']
    track_name = track_id['name']
    # This assumes one artist name, but its a list for collabs
    artist_name = track_id['artists'][0]['name']

    print(f"{idx}.) {track_name} by {artist_name}")


taylor_swift = sp.artist("06HL4z0CvFAxyc27GXpf02")
taylor_swift

taylor_albums = sp.artist_albums(taylor_swift['id'],limit=50)

for album in taylor_albums['items']:
    print(f"Album: {album['name']} -- ID: {album['id']}")



