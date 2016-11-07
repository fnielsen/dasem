"""wiktionary.

Description
-----------
Interface to Danish part of the different Wiktionaries.

"""


import requests


API_URL_PATTERN = "https://{}.wiktionary.org/w/api.php"


def get_nouns(languages=('da', 'de', 'en')):
    """Return Danish nouns from Wiktionary.

    The function will query the API of the different language versions of
    Wiktionary getting members of the (main) category of Danish nouns.

    Parameters
    ----------
    languages : list of str
        List or set of string with indication of language version of
        Wiktionary from where to query the Danish nouns from.

    Returns
    -------
    nouns : set or str
        Set of strings with Danish nouns.

    Examples
    --------
    >>> nouns = get_nouns(languages=['de'])
    >>> 'abefest' in nouns
    True

    """
    categories = {
        'da': u'Kategori:Substantiver_p\xe5_dansk',
        'de': u'Kategorie:Substantiv (D\xe4nisch)',
        'en': 'Category:Danish_nouns'
    }

    params = {
        'format': 'json',
        'action': 'query',
        'list': 'categorymembers',
        'cmlimit': 500
    }

    nouns = []
    for language in languages:
        url = API_URL_PATTERN.format(language)
        params['cmtitle'] = categories[language]
        params['cmcontinue'] = ''
        response = requests.get(url, params=params).json()
        nouns.extend([member['title']
                      for member in response['query']['categorymembers']
                      if ':' not in member['title']])
        while 'continue' in response:
            params['cmcontinue'] = response['continue']['cmcontinue']
            response = requests.get(url, params=params).json()
            nouns.extend(
                [member['title']
                 for member in response['query']['categorymembers']
                 if ':' not in member['title']])

    return set(nouns)
