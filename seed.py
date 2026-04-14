import sqlite3
import random
from datetime import datetime, timedelta

def get_sample_movies():
    """Return comprehensive sample movie data with 200 films across categories"""
    return [
        # Hollywood Movies (100 films - 50%)
        {"title": "The Dark Knight", "genre": "Action", "year": 2008, "watched": False, "rating": 0},
        {"title": "Inception", "genre": "Sci-Fi", "year": 2010, "watched": False, "rating": 0},
        {"title": "The Godfather", "genre": "Crime", "year": 1972, "watched": False, "rating": 0},
        {"title": "Pulp Fiction", "genre": "Crime", "year": 1994, "watched": False, "rating": 0},
        {"title": "The Shawshank Redemption", "genre": "Drama", "year": 1994, "watched": False, "rating": 0},
        {"title": "Forrest Gump", "genre": "Drama", "year": 1994, "watched": False, "rating": 0},
        {"title": "The Matrix", "genre": "Action", "year": 1999, "watched": False, "rating": 0},
        {"title": "Goodfellas", "genre": "Crime", "year": 1990, "watched": False, "rating": 0},
        {"title": "Fight Club", "genre": "Drama", "year": 1999, "watched": False, "rating": 0},
        {"title": "The Lord of the Rings: The Fellowship of the Ring", "genre": "Adventure", "year": 2001, "watched": False, "rating": 0},
        {"title": "Star Wars: Episode V - The Empire Strikes Back", "genre": "Sci-Fi", "year": 1980, "watched": False, "rating": 0},
        {"title": "The Silence of the Lambs", "genre": "Thriller", "year": 1991, "watched": False, "rating": 0},
        {"title": "Schindler's List", "genre": "Drama", "year": 1993, "watched": False, "rating": 0},
        {"title": "Interstellar", "genre": "Sci-Fi", "year": 2014, "watched": False, "rating": 0},
        {"title": "The Departed", "genre": "Crime", "year": 2006, "watched": False, "rating": 0},
        {"title": "Gladiator", "genre": "Action", "year": 2000, "watched": False, "rating": 0},
        {"title": "The Prestige", "genre": "Drama", "year": 2006, "watched": False, "rating": 0},
        {"title": "The Green Mile", "genre": "Drama", "year": 1999, "watched": False, "rating": 0},
        {"title": "American History X", "genre": "Drama", "year": 1998, "watched": False, "rating": 0},
        {"title": "The Usual Suspects", "genre": "Crime", "year": 1995, "watched": False, "rating": 0},
        {"title": "Leon: The Professional", "genre": "Action", "year": 1994, "watched": False, "rating": 0},
        {"title": "Saving Private Ryan", "genre": "War", "year": 1998, "watched": False, "rating": 0},
        {"title": "The Lion King", "genre": "Animation", "year": 1994, "watched": False, "rating": 0},
        {"title": "Back to the Future", "genre": "Adventure", "year": 1985, "watched": False, "rating": 0},
        {"title": "The Shining", "genre": "Horror", "year": 1980, "watched": False, "rating": 0},
        {"title": "Django Unchained", "genre": "Western", "year": 2012, "watched": False, "rating": 0},
        {"title": "The Godfather Part II", "genre": "Crime", "year": 1974, "watched": False, "rating": 0},
        {"title": "Alien", "genre": "Horror", "year": 1979, "watched": False, "rating": 0},
        {"title": "Avengers: Endgame", "genre": "Action", "year": 2019, "watched": False, "rating": 0},
        {"title": "The Dark Knight Rises", "genre": "Action", "year": 2012, "watched": False, "rating": 0},
        {"title": "Inglourious Basterds", "genre": "War", "year": 2009, "watched": False, "rating": 0},
        {"title": "The Wolf of Wall Street", "genre": "Biography", "year": 2013, "watched": False, "rating": 0},
        {"title": "Dune", "genre": "Sci-Fi", "year": 2021, "watched": False, "rating": 0},
        {"title": "Joker", "genre": "Crime", "year": 2019, "watched": False, "rating": 0},
        {"title": "Parasite", "genre": "Thriller", "year": 2019, "watched": False, "rating": 0},
        {"title": "La La Land", "genre": "Musical", "year": 2016, "watched": False, "rating": 0},
        {"title": "Whiplash", "genre": "Drama", "year": 2014, "watched": False, "rating": 0},
        {"title": "The Social Network", "genre": "Biography", "year": 2010, "watched": False, "rating": 0},
        {"title": "No Country for Old Men", "genre": "Crime", "year": 2007, "watched": False, "rating": 0},
        {"title": "There Will Be Blood", "genre": "Drama", "year": 2007, "watched": False, "rating": 0},
        {"title": "Eternal Sunshine of the Spotless Mind", "genre": "Romance", "year": 2004, "watched": False, "rating": 0},
        {"title": "The Truman Show", "genre": "Comedy", "year": 1998, "watched": False, "rating": 0},
        {"title": "Good Will Hunting", "genre": "Drama", "year": 1997, "watched": False, "rating": 0},
        {"title": "Fargo", "genre": "Crime", "year": 1996, "watched": False, "rating": 0},
        {"title": "Heat", "genre": "Action", "year": 1995, "watched": False, "rating": 0},
        {"title": "The Sixth Sense", "genre": "Thriller", "year": 1999, "watched": False, "rating": 0},
        {"title": "The Big Lebowski", "genre": "Comedy", "year": 1998, "watched": False, "rating": 0},
        {"title": "Reservoir Dogs", "genre": "Crime", "year": 1992, "watched": False, "rating": 0},
        {"title": "Blade Runner", "genre": "Sci-Fi", "year": 1982, "watched": False, "rating": 0},
        {"title": "Apocalypse Now", "genre": "War", "year": 1979, "watched": False, "rating": 0},
        {"title": "Taxi Driver", "genre": "Crime", "year": 1976, "watched": False, "rating": 0},
        {"title": "One Flew Over the Cuckoo's Nest", "genre": "Drama", "year": 1975, "watched": False, "rating": 0},
        {"title": "The Exorcist", "genre": "Horror", "year": 1973, "watched": False, "rating": 0},
        {"title": "A Clockwork Orange", "genre": "Crime", "year": 1971, "watched": False, "rating": 0},
        {"title": "2001: A Space Odyssey", "genre": "Sci-Fi", "year": 1968, "watched": False, "rating": 0},
        {"title": "Psycho", "genre": "Horror", "year": 1960, "watched": False, "rating": 0},
        {"title": "Casablanca", "genre": "Drama", "year": 1942, "watched": False, "rating": 0},
        {"title": "Citizen Kane", "genre": "Drama", "year": 1941, "watched": False, "rating": 0},
        {"title": "Gone with the Wind", "genre": "Drama", "year": 1939, "watched": False, "rating": 0},
        {"title": "The Wizard of Oz", "genre": "Adventure", "year": 1939, "watched": False, "rating": 0},
        {"title": "Metropolis", "genre": "Sci-Fi", "year": 1927, "watched": False, "rating": 0},
        {"title": "Mad Max: Fury Road", "genre": "Action", "year": 2015, "watched": False, "rating": 0},
        {"title": "John Wick", "genre": "Action", "year": 2014, "watched": False, "rating": 0},
        {"title": "Guardians of the Galaxy", "genre": "Action", "year": 2014, "watched": False, "rating": 0},
        {"title": "Deadpool", "genre": "Action", "year": 2016, "watched": False, "rating": 0},
        {"title": "Logan", "genre": "Action", "year": 2017, "watched": False, "rating": 0},
        {"title": "Black Panther", "genre": "Action", "year": 2018, "watched": False, "rating": 0},
        {"title": "Spider-Man: Into the Spider-Verse", "genre": "Animation", "year": 2018, "watched": False, "rating": 0},
        {"title": "Toy Story", "genre": "Animation", "year": 1995, "watched": False, "rating": 0},
        {"title": "Finding Nemo", "genre": "Animation", "year": 2003, "watched": False, "rating": 0},
        {"title": "The Incredibles", "genre": "Animation", "year": 2004, "watched": False, "rating": 0},
        {"title": "Shrek", "genre": "Animation", "year": 2001, "watched": False, "rating": 0},
        {"title": "Frozen", "genre": "Animation", "year": 2013, "watched": False, "rating": 0},
        {"title": "Zootopia", "genre": "Animation", "year": 2016, "watched": False, "rating": 0},
        {"title": "Coco", "genre": "Animation", "year": 2017, "watched": False, "rating": 0},
        {"title": "Up", "genre": "Animation", "year": 2009, "watched": False, "rating": 0},
        {"title": "Wall-E", "genre": "Animation", "year": 2008, "watched": False, "rating": 0},
        {"title": "Ratatouille", "genre": "Animation", "year": 2007, "watched": False, "rating": 0},
        {"title": "The Jungle Book", "genre": "Animation", "year": 1967, "watched": False, "rating": 0},
        {"title": "Beauty and the Beast", "genre": "Animation", "year": 1991, "watched": False, "rating": 0},
        {"title": "Aladdin", "genre": "Animation", "year": 1992, "watched": False, "rating": 0},
        {"title": "Mulan", "genre": "Animation", "year": 1998, "watched": False, "rating": 0},
        {"title": "Moana", "genre": "Animation", "year": 2016, "watched": False, "rating": 0},
        {"title": "The Little Mermaid", "genre": "Animation", "year": 1989, "watched": False, "rating": 0},
        {"title": "Snow White and the Seven Dwarfs", "genre": "Animation", "year": 1937, "watched": False, "rating": 0},
        {"title": "Pinocchio", "genre": "Animation", "year": 1940, "watched": False, "rating": 0},
        {"title": "Bambi", "genre": "Animation", "year": 1942, "watched": False, "rating": 0},
        {"title": "Dumbo", "genre": "Animation", "year": 1941, "watched": False, "rating": 0},
        {"title": "Fantasia", "genre": "Animation", "year": 1940, "watched": False, "rating": 0},
        {"title": "The Hunchback of Notre Dame", "genre": "Animation", "year": 1996, "watched": False, "rating": 0},
        {"title": "Hercules", "genre": "Animation", "year": 1997, "watched": False, "rating": 0},
        {"title": "Tarzan", "genre": "Animation", "year": 1999, "watched": False, "rating": 0},
        {"title": "The Emperor's New Groove", "genre": "Animation", "year": 2000, "watched": False, "rating": 0},
        {"title": "Lilo & Stitch", "genre": "Animation", "year": 2002, "watched": False, "rating": 0},
        {"title": "Brother Bear", "genre": "Animation", "year": 2003, "watched": False, "rating": 0},
        {"title": "Home on the Range", "genre": "Animation", "year": 2004, "watched": False, "rating": 0},
        {"title": "Chicken Little", "genre": "Animation", "year": 2005, "watched": False, "rating": 0},
        {"title": "Meet the Robinsons", "genre": "Animation", "year": 2007, "watched": False, "rating": 0},
        {"title": "Bolt", "genre": "Animation", "year": 2008, "watched": False, "rating": 0},
        {"title": "The Princess and the Frog", "genre": "Animation", "year": 2009, "watched": False, "rating": 0},
        {"title": "Tangled", "genre": "Animation", "year": 2010, "watched": False, "rating": 0},
        {"title": "Wreck-It Ralph", "genre": "Animation", "year": 2012, "watched": False, "rating": 0},
        {"title": "Big Hero 6", "genre": "Animation", "year": 2014, "watched": False, "rating": 0},

        # Bollywood Movies (40 films - 20%)
        {"title": "3 Idiots", "genre": "Comedy", "year": 2009, "watched": False, "rating": 0},
        {"title": "Dangal", "genre": "Drama", "year": 2016, "watched": False, "rating": 0},
        {"title": "Lagaan", "genre": "Drama", "year": 2001, "watched": False, "rating": 0},
        {"title": "Sholay", "genre": "Action", "year": 1975, "watched": False, "rating": 0},
        {"title": "Mughal-E-Azam", "genre": "Drama", "year": 1960, "watched": False, "rating": 0},
        {"title": "Dilwale Dulhania Le Jayenge", "genre": "Romance", "year": 1995, "watched": False, "rating": 0},
        {"title": "Gully Boy", "genre": "Musical", "year": 2019, "watched": False, "rating": 0},
        {"title": "Queen", "genre": "Comedy", "year": 2014, "watched": False, "rating": 0},
        {"title": "Zindagi Na Milegi Dobara", "genre": "Comedy", "year": 2011, "watched": False, "rating": 0},
        {"title": "Andhadhun", "genre": "Thriller", "year": 2018, "watched": False, "rating": 0},
        {"title": "Barfi!", "genre": "Romance", "year": 2012, "watched": False, "rating": 0},
        {"title": "PK", "genre": "Comedy", "year": 2014, "watched": False, "rating": 0},
        {"title": "Bajrangi Bhaijaan", "genre": "Drama", "year": 2015, "watched": False, "rating": 0},
        {"title": "Kabhi Khushi Kabhie Gham", "genre": "Drama", "year": 2001, "watched": False, "rating": 0},
        {"title": "Kuch Kuch Hota Hai", "genre": "Romance", "year": 1998, "watched": False, "rating": 0},
        {"title": "Kal Ho Naa Ho", "genre": "Romance", "year": 2003, "watched": False, "rating": 0},
        {"title": "Om Shanti Om", "genre": "Drama", "year": 2007, "watched": False, "rating": 0},
        {"title": "Chennai Express", "genre": "Comedy", "year": 2013, "watched": False, "rating": 0},
        {"title": "Yeh Jawaani Hai Deewani", "genre": "Romance", "year": 2013, "watched": False, "rating": 0},
        {"title": "Bahubali: The Beginning", "genre": "Action", "year": 2015, "watched": False, "rating": 0},
        {"title": "Bahubali 2: The Conclusion", "genre": "Action", "year": 2017, "watched": False, "rating": 0},
        {"title": "Rang De Basanti", "genre": "Drama", "year": 2006, "watched": False, "rating": 0},
        {"title": "Taare Zameen Par", "genre": "Drama", "year": 2007, "watched": False, "rating": 0},
        {"title": "Swades", "genre": "Drama", "year": 2004, "watched": False, "rating": 0},
        {"title": "Devdas", "genre": "Romance", "year": 2002, "watched": False, "rating": 0},
        {"title": "Black Friday", "genre": "Crime", "year": 2004, "watched": False, "rating": 0},
        {"title": "Gangs of Wasseypur", "genre": "Crime", "year": 2012, "watched": False, "rating": 0},
        {"title": "Udaan", "genre": "Drama", "year": 2010, "watched": False, "rating": 0},
        {"title": "Masaan", "genre": "Drama", "year": 2015, "watched": False, "rating": 0},
        {"title": "Newton", "genre": "Comedy", "year": 2017, "watched": False, "rating": 0},
        {"title": "Tumbbad", "genre": "Horror", "year": 2018, "watched": False, "rating": 0},
        {"title": "Stree", "genre": "Comedy", "year": 2018, "watched": False, "rating": 0},
        {"title": "Badhaai Ho", "genre": "Comedy", "year": 2018, "watched": False, "rating": 0},
        {"title": "Article 15", "genre": "Crime", "year": 2019, "watched": False, "rating": 0},
        {"title": "Thappad", "genre": "Drama", "year": 2020, "watched": False, "rating": 0},
        {"title": "Shershaah", "genre": "War", "year": 2021, "watched": False, "rating": 0},
        {"title": "Sardar Udham", "genre": "Biography", "year": 2021, "watched": False, "rating": 0},
        {"title": "Drishyam", "genre": "Thriller", "year": 2015, "watched": False, "rating": 0},
        {"title": "Kahaani", "genre": "Thriller", "year": 2012, "watched": False, "rating": 0},
        {"title": "Vicky Donor", "genre": "Comedy", "year": 2012, "watched": False, "rating": 0},

        # Animation Movies (20 films - 10%)
        {"title": "Spirited Away", "genre": "Animation", "year": 2001, "watched": False, "rating": 0},
        {"title": "My Neighbor Totoro", "genre": "Animation", "year": 1988, "watched": False, "rating": 0},
        {"title": "Princess Mononoke", "genre": "Animation", "year": 1997, "watched": False, "rating": 0},
        {"title": "Grave of the Fireflies", "genre": "Animation", "year": 1988, "watched": False, "rating": 0},
        {"title": "Howl's Moving Castle", "genre": "Animation", "year": 2004, "watched": False, "rating": 0},
        {"title": "Your Name", "genre": "Animation", "year": 2016, "watched": False, "rating": 0},
        {"title": "A Silent Voice", "genre": "Animation", "year": 2016, "watched": False, "rating": 0},
        {"title": "Weathering With You", "genre": "Animation", "year": 2019, "watched": False, "rating": 0},
        {"title": "Perfect Blue", "genre": "Animation", "year": 1997, "watched": False, "rating": 0},
        {"title": "Paprika", "genre": "Animation", "year": 2006, "watched": False, "rating": 0},
        {"title": "The Tale of the Princess Kaguya", "genre": "Animation", "year": 2013, "watched": False, "rating": 0},
        {"title": "Wolf Children", "genre": "Animation", "year": 2012, "watched": False, "rating": 0},
        {"title": "Summer Wars", "genre": "Animation", "year": 2009, "watched": False, "rating": 0},
        {"title": "The Girl Who Leapt Through Time", "genre": "Animation", "year": 2006, "watched": False, "rating": 0},
        {"title": "Akira", "genre": "Animation", "year": 1988, "watched": False, "rating": 0},
        {"title": "Ghost in the Shell", "genre": "Animation", "year": 1995, "watched": False, "rating": 0},
        {"title": "Ninja Scroll", "genre": "Animation", "year": 1993, "watched": False, "rating": 0},
        {"title": "Perfect Blue", "genre": "Animation", "year": 1997, "watched": False, "rating": 0},
        {"title": "Millennium Actress", "genre": "Animation", "year": 2001, "watched": False, "rating": 0},
        {"title": "Tokyo Godfathers", "genre": "Animation", "year": 2003, "watched": False, "rating": 0},

        # Korean, Chinese & Japanese Movies (40 films - 20%)
        {"title": "Oldboy", "genre": "Thriller", "year": 2003, "watched": False, "rating": 0},
        {"title": "Parasite", "genre": "Thriller", "year": 2019, "watched": False, "rating": 0},
        {"title": "Train to Busan", "genre": "Horror", "year": 2016, "watched": False, "rating": 0},
        {"title": "The Handmaiden", "genre": "Drama", "year": 2016, "watched": False, "rating": 0},
        {"title": "Memories of Murder", "genre": "Crime", "year": 2003, "watched": False, "rating": 0},
        {"title": "I Saw the Devil", "genre": "Thriller", "year": 2010, "watched": False, "rating": 0},
        {"title": "A Tale of Two Sisters", "genre": "Horror", "year": 2003, "watched": False, "rating": 0},
        {"title": "The Wailing", "genre": "Horror", "year": 2016, "watched": False, "rating": 0},
        {"title": "Burning", "genre": "Drama", "year": 2018, "watched": False, "rating": 0},
        {"title": "Mother", "genre": "Drama", "year": 2009, "watched": False, "rating": 0},
        {"title": "The Host", "genre": "Horror", "year": 2006, "watched": False, "rating": 0},
        {"title": "Sympathy for Mr. Vengeance", "genre": "Crime", "year": 2002, "watched": False, "rating": 0},
        {"title": "Lady Vengeance", "genre": "Crime", "year": 2005, "watched": False, "rating": 0},
        {"title": "The Chaser", "genre": "Thriller", "year": 2008, "watched": False, "rating": 0},
        {"title": "The Man from Nowhere", "genre": "Action", "year": 2010, "watched": False, "rating": 0},
        {"title": "Crouching Tiger, Hidden Dragon", "genre": "Action", "year": 2000, "watched": False, "rating": 0},
        {"title": "In the Mood for Love", "genre": "Romance", "year": 2000, "watched": False, "rating": 0},
        {"title": "Hero", "genre": "Action", "year": 2002, "watched": False, "rating": 0},
        {"title": "House of Flying Daggers", "genre": "Action", "year": 2004, "watched": False, "rating": 0},
        {"title": "Curse of the Golden Flower", "genre": "Drama", "year": 2006, "watched": False, "rating": 0},
        {"title": "The Grandmaster", "genre": "Action", "year": 2013, "watched": False, "rating": 0},
        {"title": "Seven Samurai", "genre": "Action", "year": 1954, "watched": False, "rating": 0},
        {"title": "Rashomon", "genre": "Drama", "year": 1950, "watched": False, "rating": 0},
        {"title": "Tokyo Story", "genre": "Drama", "year": 1953, "watched": False, "rating": 0},
        {"title": "Harakiri", "genre": "Drama", "year": 1962, "watched": False, "rating": 0},
        {"title": "Yojimbo", "genre": "Action", "year": 1961, "watched": False, "rating": 0},
        {"title": "Ran", "genre": "Drama", "year": 1985, "watched": False, "rating": 0},
        {"title": "Kagemusha", "genre": "Drama", "year": 1980, "watched": False, "rating": 0},
        {"title": "Throne of Blood", "genre": "Drama", "year": 1957, "watched": False, "rating": 0},
        {"title": "Ugetsu", "genre": "Drama", "year": 1953, "watched": False, "rating": 0},
        {"title": "Ikiru", "genre": "Drama", "year": 1952, "watched": False, "rating": 0},
        {"title": "High and Low", "genre": "Crime", "year": 1963, "watched": False, "rating": 0},
        {"title": "Battle Royale", "genre": "Action", "year": 2000, "watched": False, "rating": 0},
        {"title": "Audition", "genre": "Horror", "year": 1999, "watched": False, "rating": 0},
        {"title": "Ringu", "genre": "Horror", "year": 1998, "watched": False, "rating": 0},
        {"title": "Ju-on: The Grudge", "genre": "Horror", "year": 2002, "watched": False, "rating": 0},
        {"title": "Dark Water", "genre": "Horror", "year": 2002, "watched": False, "rating": 0},
        {"title": "The Eye", "genre": "Horror", "year": 2002, "watched": False, "rating": 0},
        {"title": "Infernal Affairs", "genre": "Crime", "year": 2002, "watched": False, "rating": 0},
        {"title": "Kung Fu Hustle", "genre": "Comedy", "year": 2004, "watched": False, "rating": 0},
        {"title": "Kung asdf", "genre": "Comedy", "year": 2004, "watched": False, "rating": 0}
    ]

def seed_database(db_path="movies.db"):
    """Seed the database with sample data"""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Check if database is already seeded
        c.execute("SELECT COUNT(*) FROM enhanced_movies")
        count = c.fetchone()[0]
        
        if count > 0:
            print(f"Database already contains {count} movies. Skipping seeding.")
            conn.close()
            return False
        
        # Get sample movies
        movies = get_sample_movies()
        
        # Insert sample movies
        for movie in movies:
            c.execute('''
                INSERT INTO enhanced_movies 
                (title, genre, year, watched, rating, review, added_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now', '-' || abs(random() % 365) || ' days'))
            ''', (
                movie["title"],
                movie["genre"],
                movie["year"],
                movie["watched"],
                movie["rating"],
                f"Sample review for {movie['title']}" if movie["rating"] > 0 else ""
            ))
        
        conn.commit()
        conn.close()
        
        print(f"Successfully seeded database with {len(movies)} sample movies!")
        return True
        
    except Exception as e:
        print(f"Error seeding database: {e}")
        return False

def clear_database(db_path="movies.db"):
    """Clear all data from the database"""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute("DELETE FROM enhanced_movies")
        c.execute("DELETE FROM sqlite_sequence WHERE name='enhanced_movies'")
        
        conn.commit()
        conn.close()
        
        print("Database cleared successfully!")
        return True
        
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False

if __name__ == "__main__":
    # Seed the database when this file is run directly
    seed_database()