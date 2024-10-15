from enum import Enum


class GenderBirth(Enum):
    MALE = ("male", 1)
    FEMALE = ("female", 2)
    NULL = ("", 3)

    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value


class Ethnicity(Enum):
    NON_WHITE = ("non-white", 0)
    WHITE = ("white", 1)
    NULL = ("", 2)

    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value


class EmploymentStatus(Enum):
    WORKING = ("working", 1)
    NOT_WORKING = ("not working", 2)
    NULL = ("", 3)

    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value


class Education(Enum):
    SIXTEEN_OR_LESS = ("age 16 or less", 1)
    SEVENTEEN_TO_NINETEEN = ("age 17-19", 2)
    TWENTY_AND_MORE = ("age 20 or over", 3)
    STILL = ("still in full-time education", 4)
    NULL = ("", 5)

    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value


class Smoker(Enum):
    NON_SMOKER = ("non-smoker", 0)
    EX_SMOKER = ("ex-smoker", 1)
    SMOKER = ("smoker", 2)

    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value


class InjurySeverityScoreCategory(Enum):
    MILD = ("mild", 1)
    MODERATE = ("moderate", 2)
    MAJOR = ("major", 3)

    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value


class PenetratingInjury(Enum):
    PENETRATED_SKIN = ("yes", 1)
    DID_NOT_PENETRATE_SKIN = ("no", 0)

    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value
