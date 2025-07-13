# GitHub Setup Guide

Follow these steps to push your Loan Prediction Q&A System to GitHub:

## 1. Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in to your account
2. Click the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "loan-prediction-qa")
4. Add a description (optional)
5. Choose public or private visibility
6. Do NOT initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## 2. Push Your Local Repository to GitHub

GitHub will show you commands to push an existing repository. Run these commands in your terminal:

```bash
# Replace YOUR_USERNAME with your GitHub username and REPO_NAME with your repository name
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## 3. Update README Badge

After pushing to GitHub, update the badge URL in your README.md:

1. Open README.md
2. Replace "yourusername/loan-prediction-qa" with your actual GitHub username and repository name
3. Commit and push the changes:

```bash
git add README.md
git commit -m "Update README badge"
git push
```

## 4. Set Up GitHub Pages (Optional)

To create a simple website for your project:

1. Go to your repository on GitHub
2. Click "Settings"
3. Scroll down to "GitHub Pages" section
4. Select "main" branch and "/docs" folder
5. Click "Save"

## 5. Additional GitHub Features to Consider

- **Issues**: Track bugs and feature requests
- **Projects**: Manage work with Kanban boards
- **Actions**: Set up CI/CD workflows (already configured in .github/workflows)
- **Discussions**: Create a community forum for your project
- **Wiki**: Add detailed documentation

## 6. Collaborating with Others

- Add collaborators in repository settings
- Use branches and pull requests for changes
- Review code before merging

## 7. Keeping Your Repository Updated

```bash
# Always pull the latest changes before starting work
git pull origin main

# Create a branch for your changes
git checkout -b feature/your-feature-name

# After making changes, push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
``` 