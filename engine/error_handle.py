class InvalidField(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

if __name__ == "__main__":
    raise InvalidField()