import string

def ucitaj_i_broji_rijeci(datoteka):
    rijecnik = {}

    try:
        with open(datoteka) as file:
            for linija in file:
                linija = linija.lower().replace(",","")
                rijeci = linija.split()

                for rijec in rijeci:
                    rijecnik[rijec] = rijecnik.get(rijec, 0) + 1

        return rijecnik

    except FileNotFoundError:
        print(f"Greška: Datoteka '{datoteka}' nije pronađena.")
        return None

if __name__ == "__main__":
    naziv_datoteke = "song.txt"
    rijecnik = ucitaj_i_broji_rijeci(naziv_datoteke)

    if rijecnik:
        
        rijeci_jednom = [rijec for rijec, broj in rijecnik.items() if broj == 1]

        print(f"Ukupan broj različitih riječi: {len(rijecnik)}")
        print(f"Broj riječi koje se pojavljuju samo jednom: {len(rijeci_jednom)}")
        print("Riječi koje se pojavljuju samo jednom:")
        print(", ".join(rijeci_jednom))
