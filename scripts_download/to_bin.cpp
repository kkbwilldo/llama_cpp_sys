#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include "json.hpp"

namespace {

struct Answers {
    std::vector<std::string> answers;
    std::vector<int> labels;

    void serialize(std::ostream& out) const {
        uint32_t n = answers.size();
        out.write((char *)&n, sizeof(n));
        for (auto& a : answers) {
            uint32_t m = a.size();
            out.write((char *)&m, sizeof(m));
            out.write(a.data(), m);
        }
        out.write((char *)labels.data(), labels.size() * sizeof(int));
    }

    bool deserialize(std::istream& in) {
        int n;
        in.read((char *)&n, sizeof(n));
        if (in.fail() || n < 0) {
            return false;
        }
        answers.resize(n);
        labels.resize(n);
        for (auto& a : answers) {
            uint32_t m;
            in.read((char *)&m, sizeof(m));
            a.resize(m);
            in.read((char *)a.data(), m);
        }
        in.read((char *)labels.data(), n * sizeof(int));
        return !in.fail();
    }

    void fromJson(const nlohmann::json& j) {
        for (auto& elem : j["answers"]) {
            answers.push_back(elem.get<std::string>());
        }
        for (auto& elem : j["labels"]) {
            labels.push_back(elem.get<int>());
        }
    }
};

struct MultiplChoice {
    std::string question;
    Answers singleCorrect;
    Answers multipleCorrect;

    void serialize(std::ostream& out) const {
        uint32_t n = question.size();
        out.write((char *)&n, sizeof(n));
        out.write(question.data(), n);
        singleCorrect.serialize(out);
        multipleCorrect.serialize(out);
    }

    bool deserialize(std::istream& in) {
        uint32_t n;
        in.read((char *)&n, sizeof(n));
        if (in.fail() || n < 0) {
            return false;
        }
        question.resize(n);
        in.read((char *)question.data(), n);
        return singleCorrect.deserialize(in) && multipleCorrect.deserialize(in);
    }

    void fromJson(const nlohmann::json& j) {
        question = j["question"].get<std::string>();
        singleCorrect.fromJson(j["single_correct"]);
        multipleCorrect.fromJson(j["multiple_correct"]);
    }
};

void serialize(std::ostream& out, const std::vector<MultiplChoice>& data) {
    uint32_t n = data.size();
    out.write((char *)&n, sizeof(n));
    if (data.empty()) return;
    std::vector<uint32_t> pos(data.size(), 0);
    out.write((char *)pos.data(), pos.size() * sizeof(pos[0]));
    int i = 0;
    for (auto& d : data) {
        pos[i++] = out.tellp();
        d.serialize(out);
    }
    out.seekp(sizeof(n), std::ios::beg);
    out.write((char *)pos.data(), pos.size() * sizeof(pos[0]));
}

void encode(const char* jsonFile, const char* binFile) {
    std::ifstream jsonIn(jsonFile);
    nlohmann::json jsonData;
    jsonIn >> jsonData;

    std::vector<MultiplChoice> data;
    for (auto& elem : jsonData) {
        MultiplChoice mc;
        mc.fromJson(elem);
        data.push_back(mc);
    }

    std::ofstream binOut(binFile, std::ios::binary);
    serialize(binOut, data);
}
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s input.json output.bin\n", argv[0]);
        return 1;
    }

    encode(argv[1], argv[2]);
    printf("Transformation complete. Data saved to %s\n", argv[2]);
    return 0;
}
