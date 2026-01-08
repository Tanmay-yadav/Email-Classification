<<<<<<< HEAD
import pandas as pd
import random

def generate_dataset(n=500):
    data = []
    
    # --- Component Lists for Job Applications ---
    names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Jamie"]
    roles = ["Data Scientist", "Project Manager", "Software Engineer", "Sales Lead"]
    skills = ["Python", "Strategic Planning", "React", "Cloud Computing"]
    
    for _ in range(n):
        name = random.choice(names)
        role = random.choice(roles)
        skill = random.choice(skills)
        body = f"Dear Hiring Manager,\n\nI am writing to apply for the {role} position. I have extensive experience in {skill} and believe I would be a great fit for your team. Please see my attached resume.\n\nBest regards,\n{name}"
        data.append({"text": body, "label": "job_application"})

    # --- Component Lists for Spam ---
    subjects = ["URGENT", "CONGRATULATIONS", "Account Alert", "Last Chance"]
    prizes = ["$1,000 Gift Card", "New iPhone 15", "Bitcoin Reward", "Free Trip"]
    actions = ["click the link", "verify your info", "claim now", "update password"]
    
    for _ in range(n):
        subject = random.choice(subjects)
        prize = random.choice(prizes)
        action = random.choice(actions)
        body = f"Subject: {subject}!! You have been selected to receive a {prize}! To receive your reward, you must {action} immediately at http://bit.ly/spam-link-{random.randint(100,999)}. Don't wait!"
        data.append({"text": body, "label": "spam"})

    return pd.DataFrame(data)

# Generate and Save
df = generate_dataset(500)
df.to_csv("email_dataset.csv", index=False)
=======
import pandas as pd
import random

def generate_dataset(n=500):
    data = []
    
    # --- Component Lists for Job Applications ---
    names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Jamie"]
    roles = ["Data Scientist", "Project Manager", "Software Engineer", "Sales Lead"]
    skills = ["Python", "Strategic Planning", "React", "Cloud Computing"]
    
    for _ in range(n):
        name = random.choice(names)
        role = random.choice(roles)
        skill = random.choice(skills)
        body = f"Dear Hiring Manager,\n\nI am writing to apply for the {role} position. I have extensive experience in {skill} and believe I would be a great fit for your team. Please see my attached resume.\n\nBest regards,\n{name}"
        data.append({"text": body, "label": "job_application"})

    # --- Component Lists for Spam ---
    subjects = ["URGENT", "CONGRATULATIONS", "Account Alert", "Last Chance"]
    prizes = ["$1,000 Gift Card", "New iPhone 15", "Bitcoin Reward", "Free Trip"]
    actions = ["click the link", "verify your info", "claim now", "update password"]
    
    for _ in range(n):
        subject = random.choice(subjects)
        prize = random.choice(prizes)
        action = random.choice(actions)
        body = f"Subject: {subject}!! You have been selected to receive a {prize}! To receive your reward, you must {action} immediately at http://bit.ly/spam-link-{random.randint(100,999)}. Don't wait!"
        data.append({"text": body, "label": "spam"})

    return pd.DataFrame(data)

# Generate and Save
df = generate_dataset(500)
df.to_csv("email_dataset.csv", index=False)
>>>>>>> ca3e75404bccb1ead8f5dbf4653b777983a7dc82
print("Dataset with 1,000 rows created successfully!")