# app/scraper.py
import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException
from datetime import datetime


async def get_upcoming_events():
    events = []
    today = datetime.today().date()
    base_url = "https://www.nparks.gov.sg/sbg/whats-happening/calendar-of-events?page={}"

    try:
        page_number = 1
        while True:
            url = base_url.format(page_number)
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            event_items = soup.find_all(class_="events__item")

            if not event_items:
                break

            for event in event_items:
                title = event.find(class_="events__title").get_text(
                    strip=True) if event.find(class_="events__title") else "No Title"
                date_text = event.find(class_="events__subsection").get_text(strip=True).replace(
                    "Date", "").strip() if event.find(class_="events__subsection") else "No Date"
                description = event.find("a", class_="events__link").get_text(
                    strip=True) if event.find("a", class_="events__link") else "No Description"

                start_date_str = date_text.split(" - ")[0]
                try:
                    event_date = datetime.strptime(
                        start_date_str, "%d %b %Y").date()
                    if event_date >= today:
                        events.append(
                            {"title": title, "date": start_date_str, "description": description})
                except ValueError:
                    continue

            pagination_items = soup.find('ul', class_='pagination')
            if pagination_items:
                next_page_item = pagination_items.find_all('li')[-1]
                if next_page_item.find('a') and 'href' in next_page_item.find('a').attrs:
                    next_page_url = next_page_item.find('a')['href']
                    if 'page' in next_page_url:
                        page_number += 1
                    else:
                        break
                else:
                    break
            else:
                break

    except requests.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code,
                            detail=f"HTTP error occurred: {http_err}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred while fetching events: {str(e)}")

    return {"upcoming_events": events} if events else {"message": "No upcoming events found."}
