import requests

def get_definition(word):
    """
    Fetches a short definition for the given word using the free
    dictionaryapi.dev JSON API (no scraping needed, no HTML to break).
    """
    word = word.strip().lower()
    if not word:
        return "No word provided."

    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return f"No definition found for '{word}'."

        data = response.json()
        meanings = data[0].get("meanings", [])
        if not meanings:
            return f"No definition found for '{word}'."

        part_of_speech = meanings[0].get("partOfSpeech", "")
        definition = meanings[0]["definitions"][0].get("definition", "")

        return f"({part_of_speech}) {definition}"

    except requests.exceptions.RequestException:
        return "Could not reach dictionary service (check internet connection)."
    except (KeyError, IndexError, ValueError):
        return f"No definition found for '{word}'."

if __name__ == "__main__":
    test_word = input("Enter a word to look up: ")
    print(get_definition(test_word))