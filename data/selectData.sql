use topcoder;
SELECT challengeId as TaskId , prize, postingDate, submissionEndDate, challengeType, technology, platforms FROM challenge_item WHERE currentStatus="Completed";
SELECT challengeID as TaskId, handle, placement, submissionDate as subDate, submissionStatus as subStatus, finalScore FROM challenge_submission group by challengeID,handle;
SELECT challengeID as TaskId, handle, registrationDate as regDate, rating FROM challenge_registrant group by challengeID,handle;
SELECT handle, memberSince from user;