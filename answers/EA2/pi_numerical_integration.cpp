#include <iomanip>
#include <iostream>
#include <omp.h>

using namespace std;

int main() {
  int num_steps = 100000000; // Anzahl der Rechtecke
  double width = 1.0 / double(num_steps); // Breite eines Rechtecks
  double sum = 0.0; // Summe der Höhen der Rechtecke

  double start_time = omp_get_wtime(); // Startzeit

  #pragma omp parallel
  {
    double sum_local = 0.0; // Lokale Summe für jeden Thread

    #pragma omp for
    for (int i = 0; i < num_steps; i++) {
      double x = (i + 0.5) * width; // Mittelpunkt
      sum_local += (1.0 / (1.0 + x * x)); // Höhe hinzufügen
    }

    #pragma omp critical
    sum += sum_local; // Globale Summe aktualisieren
  }

  double pi = sum * 4 * width; // Pi berechnen
  double run_time = omp_get_wtime() - start_time; // Laufzeit berechnen

  cout << "pi with " << num_steps << " steps is " << setprecision(17)
       << pi << " in " << setprecision(6) << run_time << " seconds\n";
}
