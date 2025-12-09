# Caramanta Website

Downloaded from: https://studiomios.wixstudio.com/caramanta

## What's Here

This is a scraped copy of the Wix Studio site. It includes:
- Main HTML page (`index.html`)
- JavaScript libraries (React, lodash, Wix components)
- Fonts (Google Fonts, Wix fonts)
- Images and media assets

## Viewing Locally

Open `index.html` in your browser:

```bash
open index.html
```

Or start a simple HTTP server:

```bash
python3 -m http.server 8000
# Then visit http://localhost:8000
```

## Limitations

- Some CSS files failed to download (likely 404s from Wix)
- The site is heavily JavaScript-dependent (Wix Thunderbolt framework)
- Some dynamic features may not work offline
- External API calls will fail

## Next Steps

To make this truly editable and deployable:

1. **Extract the actual content** (text, images, layout)
2. **Rebuild as a static site** using:
   - Plain HTML/CSS/JavaScript
   - Or a static site generator (Jekyll, 11ty, Hugo)
3. **Replace Wix dependencies** with modern alternatives
4. **Version control** and deploy to GitHub Pages, Netlify, etc.

## Files

- `index.html` - Main page
- `assets/` - Downloaded resources (JS, CSS, fonts, images)
- `scrape_wix.py` - Script used to download this site
