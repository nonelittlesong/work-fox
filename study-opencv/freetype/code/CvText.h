//
// Created by song on 18-9-27.
//

#ifndef FACEATTRIBUTES_CVTEXT_H
#define FACEATTRIBUTES_CVTEXT_H

#include <opencv2/opencv.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H

class CvText {
public:

    /**
     * 构造函数，初始化一个字体
     * @param fontName 字体名称
     */
    explicit CvText(const char *fontName);

    virtual ~CvText();

    /**
     * 设置字体属性，属性为空时保持默认值
     * @param type 类型
     * @param size 大小
     * @param underline 下划线
     * @param diaphaneity 透明度
     */
    void setFont(int *type, cv::Scalar *size = nullptr,
                 bool *underline = nullptr, float *diaphaneity = nullptr);

    /**
     * 恢复默认字体设置
     */
    void restoreFont();

    /**
     * 将text的内容放到frame的指定位置(pos)，默认文本颜色为黑色。遇到不能输出的字符将停止。
     * @param frame 输出的帧
     * @param text 文本内容
     * @param pos 文本位置
     * @param color 文本颜色
     * @return 返回字符长度，失败返回-1
     */
    int putText(cv::Mat &frame, std::string text, cv::Point pos,
                cv::Scalar color = cv::Scalar(0, 0, 0));
    int putText(cv::Mat &frame, const char *text, cv::Point pos,
                cv::Scalar color = cv::Scalar(0, 0, 0));

private:

    /**
     * 输出wc到frame的pos位置
     * @param frame 输出的Mat
     * @param wc 字符
     * @param pos 位置
     * @param color 颜色
     */
    void putWChar(cv::Mat &frame, wchar_t wc, cv::Point &pos, cv::Scalar color);

    /**
     * 将char字符数组转换为wchar_t字符数组
     * @param src char字符数组
     * @param dst wchar_t字符数组
     * @param locale 语言环境，mbstowcs函数以来此值来判断src的编码方式
     * @return 成功返回0, 失败-1
     */
    int char2Wchar(const char *&src, wchar_t *&dst, const char *locale = "zh_CN.utf8");

    FT_Library m_library;
    FT_Face m_face;

    int m_fontType;
    cv::Scalar m_fontSize;
    bool m_fontUnderline;
    float m_fontDiaphaneity;
};
#endif //FACEATTRIBUTES_CVTEXT_H
