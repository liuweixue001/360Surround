#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_surround.h"
#include "qstring.h"
#include "qfiledialog.h"
#include "qdebug.h"
#include "qimage.h"
#include "iostream"
#include "opencv2/imgproc/types_c.h"
#include "qspinbox.h"

class surround : public QMainWindow
{
    Q_OBJECT

public:
    surround(QWidget *parent = Q_NULLPTR);
    QString filepath;
private:
    Ui::surroundClass *ui;
};
