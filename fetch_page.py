import requests
import sys
from markdownify import markdownify as md

def fetch_webpage_as_markdown(url, output_filename="output.md"):
    """
    Fetches the content of a webpage and saves it as a Markdown file.

    Args:
        url (str): The URL of the webpage to fetch.
        output_filename (str): The name of the output Markdown file.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        html_content = response.text

        markdown_content = md(html_content)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Webpage content saved to {output_filename} as Markdown.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

########################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_page.py <target_url> [output_filename]")
        sys.exit(1)
    target_url = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else "output.md"
    fetch_webpage_as_markdown(target_url, output_filename)

