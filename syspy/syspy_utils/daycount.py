def count_by_weekday(start_date, end_date):

    """
    returns a list of length 7 with the number of occurrences of each day over a given period

    :param start_date: first day of the period 
    :type start_date:  datetime.datetime
    :param end_date: first day after the period 
    :type end_date:  datetime.datetime
    :return: inner_count_by_weekday
    :rtype: list
    """
    
    delta = end_date - start_date
    inner_count_by_weekday = [delta.days//7]*7

    for i in range(delta.days % 7):
        inner_count_by_weekday[(start_date.weekday() + i) % 7] += 1
        
    return inner_count_by_weekday


def count_from_mask(start_date, end_date, mask=(1, 1, 1, 1, 1, 1, 1)):

    """
    returns the number of days in a period that are consistent with the provided mask

    :param start_date: first day of the period 
    :type start_date:  datetime.datetime
    :param end_date: first day after the period 
    :type end_date:  datetime.datetime
    :param mask: boolean list of length 7 that states wich weekday should be taken into account
    :type mask: list
    :return: count
    :rtype: int
    """

    inner_count_by_weekday = count_by_weekday(start_date, end_date)
    count = 0
    for i in range(7):
        count += mask[i] * inner_count_by_weekday[i]
    return count