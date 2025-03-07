def unos_ocjene(ocjena):
    if ocjena >= 0.9:
        return "A"
    elif ocjena >= 0.8:
        return "B"
    elif ocjena >= 0.7:
        return "C"
    elif ocjena >= 0.6:
        return "D"
    else:
        return "F"

if __name__ == "__main__":
    while True:
        try:
            unos = input("Unesite ocjenu (0.0 - 1.0): ")

            if unos.lower() == 'q':
                print("Izlaz iz programa.")
                break 

            ocjena = float(unos)

            if 0.0 <= ocjena <= 1.0:
                print(f"Vaša ocjena je: {ocjena} → {unos_ocjene(ocjena)}")
                break
            else:
                print("Greška: Uneseni broj nije u rasponu od 0.0 do 1.0.")

        except ValueError:
            print("Greška: Molimo unesite broj između 0.0 i 1.0.")

