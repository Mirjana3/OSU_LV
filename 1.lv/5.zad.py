def obradi_sms_podatke(datoteka):
    try:
        with open(datoteka, encoding='utf-8') as file:
            ham_broj_rijeci, spam_broj_rijeci = 0, 0
            ham_broj_poruka, spam_broj_poruka = 0, 0
            spam_zavrsava_usklicnikom = 0

            for linija in file:
                podaci = linija.strip().split("\t")  
                
                if len(podaci) < 2:
                    continue 

                tip_poruke, tekst = podaci[0], podaci[1]
                broj_rijeci = len(tekst.split())

                if tip_poruke == "ham":
                    ham_broj_rijeci += broj_rijeci
                    ham_broj_poruka += 1
                elif tip_poruke == "spam":
                    spam_broj_rijeci += broj_rijeci
                    spam_broj_poruka += 1
                    if tekst.endswith("!"):
                        spam_zavrsava_usklicnikom += 1

            
            prosjek_ham = ham_broj_rijeci / ham_broj_poruka if ham_broj_poruka > 0 else 0
            prosjek_spam = spam_broj_rijeci / spam_broj_poruka if spam_broj_poruka > 0 else 0

            print(f"Prosječan broj riječi u ham porukama: {prosjek_ham:.2f}")
            print(f"Prosječan broj riječi u spam porukama: {prosjek_spam:.2f}")
            print(f"Broj spam poruka koje završavaju uskličnikom: {spam_zavrsava_usklicnikom}")

    except FileNotFoundError:
        print(f"Greška: Datoteka '{datoteka}' nije pronađena.")

if __name__ == "__main__":
    naziv_datoteke = "SMSSpamCollection.txt"
    obradi_sms_podatke(naziv_datoteke)