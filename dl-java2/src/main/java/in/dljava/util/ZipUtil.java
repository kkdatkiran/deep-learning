package in.dljava.util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import in.dljava.optimizer.Tuple2;
import in.dljava.optimizer.Tuple3;

public class ZipUtil {

	public static <A, B, C> List<Tuple3<A, B, C>> zip(Iterable<A> itA, Iterable<B> itB, Iterable<C> itC) {

		Iterator<A> ia = itA.iterator();
		Iterator<B> ib = itB.iterator();
		Iterator<C> ic = itC.iterator();

		return zip(ia, ib, ic);
	}

	public static <A, B, C> List<Tuple3<A, B, C>> zip(Iterator<A> ia, Iterator<B> ib, Iterator<C> ic) {

		List<Tuple3<A, B, C>> retList = new ArrayList<>();

		while (ia.hasNext() && ib.hasNext() && ic.hasNext())
			retList.add(new Tuple3<>(ia.next(), ib.next(), ic.next()));

		return retList;
	}

	public static <A, B> List<Tuple2<A, B>> zip(Iterable<A> itA, Iterable<B> itB) {

		Iterator<A> ia = itA.iterator();
		Iterator<B> ib = itB.iterator();

		return zip(ia, ib);
	}

	public static <A, B> List<Tuple2<A, B>> zip(Iterator<A> ia, Iterator<B> ib) {

		List<Tuple2<A, B>> retList = new ArrayList<>();

		while (ia.hasNext() && ib.hasNext())
			retList.add(new Tuple2<>(ia.next(), ib.next()));

		return retList;
	}

	private ZipUtil() {

	}
}
