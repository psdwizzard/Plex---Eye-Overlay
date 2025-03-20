# Setting Up Plex API Integration

The "Process Plex Server" feature allows you to connect directly to your Plex Media Server using the official Plex API. This guide will help you set up the connection.

## Required Information

To connect to your Plex server, you'll need:

1. **Your Plex server URL**: This is the address of your Plex server, including the port (usually 32400)
2. **Your Plex authentication token**: A special access token that allows the app to communicate with your Plex server

## Finding Your Plex Server URL

The Plex server URL typically follows this format:
```
http://[IP address or hostname]:32400
```

Examples:
- `http://localhost:32400` (if running on the same machine)
- `http://192.168.1.100:32400` (if running on your local network)
- `https://plex.example.com:32400` (if using a domain with SSL)

## Finding Your Plex Authentication Token

There are several ways to find your Plex authentication token:

### Method 1: From Plex Web App

1. Sign in to Plex Web App
2. Play any media item
3. While media is playing, right-click and select "Get Info" or press `I`
4. In the XML file that opens, find the `X-Plex-Token` parameter (it looks like a long string of letters and numbers)

### Method 2: From Plex Web App URL

1. Sign in to Plex Web App
2. Press F12 to open developer tools 
3. Go to the Network tab
4. Look for any request to your Plex server
5. Find the `X-Plex-Token` parameter in the request URL

### Method 3: From Plex Web App Cookies

1. Sign in to Plex Web App
2. Open browser developer tools (F12)
3. Go to the Application tab (or Storage in Firefox)
4. Look for Cookies in the left sidebar
5. Find the cookie named `X-Plex-Token`

### Method 4: From Configuration Files

Depending on your OS, you can find the token in Plex configuration files:

**Windows**:
```
%LOCALAPPDATA%\Plex Media Server\Preferences.xml
```

**Mac**:
```
~/Library/Application Support/Plex Media Server/Preferences.xml
```

**Linux**:
```
$PLEX_HOME/Library/Application Support/Plex Media Server/Preferences.xml
```

Look for an attribute called `PlexOnlineToken` in the XML file.

## Library Selection

After connecting to your Plex server, you'll see a list of available libraries. Select the libraries you want to process:

- **Movie Libraries**: Select the movie libraries containing posters you want to modify
- **TV Show Libraries**: Select the TV show libraries containing posters you want to modify

## Processing Options

- **Debug Mode**: When enabled, saves visualization images showing the face detection results
- **Movie Poster Mode**: Provides better detection for angled faces commonly found in movie posters

## Limitations and Considerations

- Processed posters will overwrite the existing posters on your Plex server
- The operation cannot be undone (make backups if needed)
- Processing may take some time depending on the size of your libraries
- For each item, the app needs to:
  1. Download the poster
  2. Process it to add googly eyes
  3. Upload the modified poster back to Plex
- Network speed and CPU power will affect processing time 