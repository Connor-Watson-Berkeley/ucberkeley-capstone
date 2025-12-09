#!/usr/bin/env python3
"""
Generate complete static website from extracted Wix content
"""
import json

# Load extracted content
with open('content.json', 'r', encoding='utf-8') as f:
    content = json.load(f)

# Common header/footer
def header(page_name):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_name} - Caramanta Data Platform</title>
    <link rel="stylesheet" href="clean/style.css">
</head>
<body>
    <header class="header">
        <nav class="nav">
            <div class="container">
                <div class="nav-content">
                    <div class="logo">Caramanta</div>
                    <ul class="nav-menu">
                        <li><a href="home.html" {"class=\"active\"" if page_name == "Home" else ""}>Home</a></li>
                        <li><a href="problem.html" {"class=\"active\"" if page_name == "Problem" else ""}>Problem</a></li>
                        <li><a href="solution.html" {"class=\"active\"" if page_name == "Solution" else ""}>Solution</a></li>
                        <li><a href="team.html" {"class=\"active\"" if page_name == "Team" else ""}>Team</a></li>
                    </ul>
                </div>
            </div>
        </nav>
    </header>
'''

def footer():
    return '''
    <footer class="footer">
        <div class="container">
            <div class="footer-grid">
                <div class="footer-col">
                    <h3>GET PROJECT UPDATES</h3>
                    <form class="newsletter-form">
                        <input type="email" placeholder="Enter your email" required>
                        <button type="submit" class="btn btn-small">Subscribe</button>
                    </form>
                </div>
                <div class="footer-col">
                    <h3>QUICK LINKS</h3>
                    <ul class="footer-links">
                        <li><a href="home.html">Home</a></li>
                        <li><a href="problem.html">The Problem</a></li>
                        <li><a href="solution.html">Solution</a></li>
                        <li><a href="team.html">Team</a></li>
                    </ul>
                </div>
                <div class="footer-col">
                    <h3>CONTACT</h3>
                    <p>Info@mysite.com</p>
                    <p>123-456-7890</p>
                    <p>500 Terry Francine Street<br>San Francisco, CA 94158</p>
                </div>
            </div>
            <div class="footer-bottom">
                <p>© 2025 by Studio MIOS.</p>
            </div>
        </div>
    </footer>
    <script src="clean/script.js"></script>
</body>
</html>'''

# Generate HOME page
home_html = header("Home")
home = content['home']
home_html += f'''
    <section class="hero">
        <div class="container">
            <div class="hero-content">
                <h1 class="hero-title">{home['h1'][0].replace(chr(10), '<br>')}</h1>
                <p class="hero-subtitle">{home['paragraphs'][0]}</p>
                <p class="hero-description">{home['paragraphs'][1]}</p>
                <a href="solution.html" class="btn btn-primary">Explore the Solution</a>
            </div>
        </div>
    </section>

    <section class="video-section">
        <div class="container">
            <div class="video-wrapper">
                <div class="video-placeholder">
                    <p>Video: Learn why this project started in Caramanta, Colombia</p>
                </div>
                <p class="video-caption">{home['paragraphs'][2]}</p>
            </div>
        </div>
    </section>

    <section class="image-banner">
        <div class="container">
            <img src="clean/images/primer-plano-dos-tipos-de-fondo-de-granos-de-cafe_edited.jpg"
                 alt="Coffee beans" class="banner-image">
            <h2 class="banner-title">{home['h2'][0].replace(chr(10), '<br>')}</h2>
        </div>
    </section>

    <section class="content-section">
        <div class="container">
            <div class="content-box">
                <h2>{home['h2'][1]}</h2>
                <p>{home['paragraphs'][3]}</p>
                <a href="problem.html" class="btn btn-secondary">Learn About the Problem</a>
            </div>
        </div>
    </section>

    <section class="content-section alt-bg">
        <div class="container">
            <div class="content-box">
                <h2>{home['h2'][2]}</h2>
                <p>{home['paragraphs'][4]}</p>
                <p class="highlight"><strong>The goal is simple:</strong> {home['paragraphs'][5]}</p>
                <a href="solution.html" class="btn btn-primary">Explore the Solution</a>
            </div>
        </div>
    </section>

    <section class="content-section">
        <div class="container">
            <div class="team-content">
                <h2>{home['h2'][3]}</h2>
                <p>{home['paragraphs'][6]}</p>
                <a href="team.html" class="btn btn-secondary">Meet the Team</a>
            </div>
        </div>
    </section>
'''
home_html += footer()

# Generate PROBLEM page
problem_html = header("Problem")
prob = content['problem']
problem_html += f'''
    <section class="hero">
        <div class="container">
            <div class="hero-content">
                <h1 class="hero-title">{prob['h1'][0]}</h1>
                <p class="hero-description">{prob['paragraphs'][0]}</p>
            </div>
        </div>
    </section>

    <section class="content-section">
        <div class="container">
            <div class="content-grid" style="max-width: 900px; margin: 0 auto;">
'''
for i, h3 in enumerate(prob['h3'][:4]):
    problem_html += f'                <h3 style="color: var(--primary-color); margin: 2rem 0 1rem;">{h3}</h3>\n'

problem_html += '''            </div>
        </div>
    </section>

    <section class="content-section alt-bg">
        <div class="container">
            <div class="content-box">
'''
problem_html += f'                <h2>{prob['h2'][0].replace(chr(10), " ")}</h2>\n'
problem_html += f'                <p class="highlight" style="font-size: 1.2rem; font-style: italic;">{prob["paragraphs"][3]}</p>\n'
problem_html += f'                <p>{prob["paragraphs"][4]}</p>\n'
problem_html += f'                <p>{prob["paragraphs"][5]}</p>\n'
problem_html += f'''                <a href="solution.html" class="btn btn-primary">See Our Solution</a>
            </div>
        </div>
    </section>
'''
problem_html += footer()

# Generate SOLUTION page
solution_html = header("Solution")
sol = content['solution']
solution_html += f'''
    <section class="hero">
        <div class="container">
            <div class="hero-content">
                <h1 class="hero-title">THE SOLUTION</h1>
                <p class="hero-subtitle">{sol['paragraphs'][1]}</p>
                <p class="hero-description">{sol['paragraphs'][0]}</p>
            </div>
        </div>
    </section>

    <section class="content-section">
        <div class="container">
            <div class="content-box">
                <h2>The Question We Answer</h2>
                <p class="highlight" style="font-size: 1.3rem; text-align: center; padding: 2rem; background: var(--bg-light); border-radius: 8px; margin: 2rem 0;">
                    {sol['paragraphs'][3]}
                </p>
                <p>{sol['paragraphs'][4]}</p>
            </div>
        </div>
    </section>

    <section class="content-section alt-bg">
        <div class="container">
            <div class="content-grid">
                <h2>{sol['h3'][0]}</h2>
                <h3 style="margin-top: 2rem;">{sol['h2'][0]}</h3>
                <ul style="list-style: disc; padding-left: 2rem; margin: 1rem 0;">
                    <li style="margin-bottom: 0.5rem;">{sol['paragraphs'][5]}</li>
                    <li style="margin-bottom: 0.5rem;">{sol['paragraphs'][6]}</li>
                    <li style="margin-bottom: 0.5rem;">{sol['paragraphs'][7]}</li>
                </ul>

                <h3 style="margin-top: 2rem;">{sol['h2'][1]}</h3>
                <ul style="list-style: disc; padding-left: 2rem; margin: 1rem 0;">
                    <li style="margin-bottom: 0.5rem;">{sol['paragraphs'][8]}</li>
                    <li style="margin-bottom: 0.5rem;">{sol['paragraphs'][9]}</li>
                    <li style="margin-bottom: 0.5rem;">{sol['paragraphs'][10]}</li>
                </ul>
            </div>
        </div>
    </section>

    <section class="content-section">
        <div class="container">
            <div class="content-box">
                <h2>{sol['h2'][2]}</h2>
                <p>{sol['paragraphs'][11]}</p>
                <p>{sol['paragraphs'][12]}</p>
            </div>
        </div>
    </section>

    <section class="content-section alt-bg">
        <div class="container">
            <div class="team-content">
                <h1 style="margin-bottom: 2rem;">{sol['h1'][0]}</h1>
                <p style="font-size: 1.2rem; max-width: 700px; margin: 0 auto;">
                    This project provides coffee producers with market intelligence that was previously available only to traders and financial institutions. By democratizing access to forecasting and risk analysis, we help create economic dignity and fairer outcomes for rural communities.
                </p>
                <a href="team.html" class="btn btn-primary" style="margin-top: 2rem;">Meet the Team</a>
            </div>
        </div>
    </section>
'''
solution_html += footer()

# Generate TEAM page
team_html = header("Team")
team = content['team']
team_html += f'''
    <section class="hero">
        <div class="container">
            <div class="hero-content" style="text-align: center;">
                <h1 class="hero-title">MEET THE TEAM</h1>
                <p class="hero-description">{team['paragraphs'][0]}</p>
            </div>
        </div>
    </section>

    <section class="content-section">
        <div class="container">
            <div class="team-content">
                <h2>Our Mission</h2>
                <p style="font-size: 1.2rem;">{team['paragraphs'][1]}</p>
            </div>
        </div>
    </section>

    <section class="content-section alt-bg">
        <div class="container">
            <h2 style="text-align: center; margin-bottom: 3rem;">{team['paragraphs'][2]}</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;">
'''

# Add team members
team_members = [
    ("Stuart Holland", "Chief Executive Officer (CEO)"),
    ("Francisco Muñoz", "Chief Financial Officer (CFO)"),
    ("Tony Gibbons", "Chief Technology Officer (CTO)"),
    ("Connor Watson", "Chief Executive Officer (CEO)")
]

for name, title in team_members:
    team_html += f'''                <div style="text-align: center; padding: 2rem; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: var(--primary-color); margin-bottom: 0.5rem;">{name}</h3>
                    <p style="color: var(--text-light); font-size: 0.9rem;">{title}</p>
                </div>
'''

team_html += '''            </div>
        </div>
    </section>

    <section class="content-section">
        <div class="container">
            <div class="team-content">
                <h2>Acknowledgments</h2>
                <p>We are deeply grateful to:</p>
                <ul style="list-style: disc; text-align: left; max-width: 600px; margin: 1rem auto; padding-left: 2rem;">
                    <li style="margin-bottom: 0.5rem;">The coffee-growing community of Caramanta</li>
                    <li style="margin-bottom: 0.5rem;">Industry experts who shared their insights</li>
                    <li style="margin-bottom: 0.5rem;">UC Berkeley School of Information faculty and staff</li>
                </ul>
            </div>
        </div>
    </section>
'''
team_html += footer()

# Write all HTML files
with open('clean/home.html', 'w', encoding='utf-8') as f:
    f.write(home_html)
with open('clean/problem.html', 'w', encoding='utf-8') as f:
    f.write(problem_html)
with open('clean/solution.html', 'w', encoding='utf-8') as f:
    f.write(solution_html)
with open('clean/team.html', 'w', encoding='utf-8') as f:
    f.write(team_html)

print("✅ Generated all HTML pages:")
print("  - home.html")
print("  - problem.html")
print("  - solution.html")
print("  - team.html")
