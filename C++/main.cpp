#include "surround.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    surround w;
    w.show();
    return a.exec();
}
