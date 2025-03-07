if __name__ == "__main__":
    brojevi = [] 

    while True:
        try:
            unos = input("Unesite broj ili 'Done' za kraj: ").strip()

            if unos.lower() == "done":  
                break

            broj = float(unos)  
            brojevi.append(broj)

        except ValueError:
            print("Greška: Unesite ispravan broj ili 'Done' za završetak.")

    if brojevi: 
        print("\nStatistika unesenih brojeva:")
        print(f"Broj unosa: {len(brojevi)}")
        print(f"Srednja vrijednost: {sum(brojevi) / len(brojevi):.2f}")
        print(f"Minimalna vrijednost: {min(brojevi)}")
        print(f"Maksimalna vrijednost: {max(brojevi)}")
        print(f"Sortirana lista: {sorted(brojevi)}")
    else:
        print("Niste unijeli niti jedan broj.")
