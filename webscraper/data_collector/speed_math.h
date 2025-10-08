int int_log10(unsigned int x) {
    int log = 1;
    while (x >= 10) {
        x /= 10;
        log++;
    }
    return log;
}