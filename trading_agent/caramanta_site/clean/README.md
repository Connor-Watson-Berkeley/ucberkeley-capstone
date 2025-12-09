# Caramanta - Clean Static Website

**Fully independent, Wix-free static website** - ready to edit, deploy, and customize.

## What's This?

This is a **clean rebuild** of the Caramanta Wix site as a fully independent static website. No Wix dependencies, no external frameworks - just clean HTML, CSS, and vanilla JavaScript.

## Features

✅ **Fully Independent**
- No Wix dependencies
- No external JavaScript frameworks
- Works offline
- Fast loading times

✅ **Modern & Responsive**
- Mobile-first design
- Smooth animations
- Responsive navigation
- Touch-friendly

✅ **Easy to Edit**
- Clean, semantic HTML
- Well-organized CSS with CSS variables
- Commented JavaScript
- Simple folder structure

✅ **SEO Ready**
- Semantic HTML5 markup
- Meta tags included
- Fast performance
- Accessible

## File Structure

```
clean/
├── index.html          # Main HTML file (fully editable)
├── style.css           # All styles (CSS variables for easy theming)
├── script.js           # Interactive features (menu, smooth scroll, etc.)
├── images/             # Image assets
│   └── *.jpg
└── README.md           # This file
```

## Quick Start

### View Locally

**Option 1: Open directly**
```bash
open index.html
```

**Option 2: Run local server (recommended)**
```bash
# Python 3
python3 -m http.server 8000

# Then visit: http://localhost:8000
```

### Edit Content

1. **Text Content:** Edit `index.html`
   - All text is in plain HTML
   - No special syntax or frameworks

2. **Styling:** Edit `style.css`
   - Colors are defined as CSS variables (`:root` section)
   - Organized by sections
   - Mobile styles at the bottom

3. **Behavior:** Edit `script.js`
   - Mobile menu toggle
   - Smooth scrolling
   - Newsletter form
   - Fade-in animations

### Customize Colors

Edit the `:root` section in `style.css`:

```css
:root {
    --primary-color: #2B5940;    /* Main brand color */
    --secondary-color: #8B4513;  /* Secondary accent */
    --accent-color: #D4A574;     /* Highlights */
    --text-dark: #1a1a1a;        /* Main text */
    --text-light: #666;          /* Secondary text */
    --bg-light: #f9f7f4;         /* Light background */
}
```

## Deploy

### GitHub Pages

1. Create a GitHub repository
2. Push this folder to the repo
3. Enable GitHub Pages in Settings
4. Your site will be live at `https://username.github.io/repo-name`

### Netlify

1. Drag and drop the `clean/` folder to Netlify
2. Your site is live instantly
3. Get a custom domain if desired

### Vercel

```bash
cd clean
vercel deploy
```

### Any Web Host

Upload the entire `clean/` folder to your web hosting via FTP/SFTP.

## What's Different from the Original Wix Site?

| Feature | Wix Site | Clean Site |
|---------|----------|------------|
| **Load Time** | ~3-5 seconds | <1 second |
| **File Size** | 3.2MB (with framework) | ~50KB |
| **Dependencies** | Wix framework required | None - fully independent |
| **Editability** | Wix editor only | Any text editor |
| **Hosting** | Wix servers only | Host anywhere |
| **Performance** | Heavy JavaScript | Lightweight vanilla JS |
| **Cost** | Wix subscription | Free (host anywhere) |

## Sections

The site includes:

1. **Hero Section** - Main headline and call-to-action
2. **Video Section** - Project introduction (placeholder)
3. **Image Banner** - Visual impact section
4. **Problem Section** - Why this matters
5. **Solution Section** - How the platform helps
6. **Team Section** - Meet the team
7. **Footer** - Newsletter signup, links, contact

## Browser Support

- ✅ Chrome (last 2 versions)
- ✅ Firefox (last 2 versions)
- ✅ Safari (last 2 versions)
- ✅ Edge (last 2 versions)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## Next Steps

1. **Add Video:** Replace the video placeholder with an actual video or YouTube embed
2. **More Images:** Add more coffee/Colombia imagery to make it richer
3. **Add Pages:** Create separate HTML files for Problem, Solution, Team pages
4. **Contact Form:** Implement a working contact form (use Formspree, Netlify Forms, etc.)
5. **Analytics:** Add Google Analytics or similar
6. **Custom Domain:** Point your domain to the hosted site

## License

© 2025 by Studio MIOS

---

**Need help?** This is a standard HTML/CSS/JS website - any web developer can help customize it further.
