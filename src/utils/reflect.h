
#include "boost/pfr.hpp"
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/bind/bind.hpp>

// dump name macro
#define DUMPSTR_NAME_VAL(os, name, a)               \
    do {                                            \
        (os) << (name) << ": " << (a) << std::endl; \
    } while (false)

#define DUMPALL(os, a) DUMPSTR_NAME_VAL((os), #a, (a))

#define DUMP_NAME(os, a)                            \
    do {                                            \
        (os) << (#a) << "," << (a)                  \
    } while (false)

// define typedef
#define REM(...) __VA_ARGS__
#define EAT(...)
// Retrieve the type
#define TYPEOF(x) DETAIL_TYPEOF(DETAIL_TYPEOF_PROBE x, )
#define DETAIL_TYPEOF(...) DETAIL_TYPEOF_HEAD(__VA_ARGS__)
#define DETAIL_TYPEOF_HEAD(x, ...) REM x
#define DETAIL_TYPEOF_PROBE(...) (__VA_ARGS__),
// Strip off the type
#define STRIP(x) EAT x
// Show the type without parenthesis
#define PAIR(x) REM x

// A helper metafunction for adding const to a type
template <class M, class T>
struct make_const
{
  typedef T type;
};

template <class M, class T>
struct make_const<const M, T>
{
  typedef typename boost::add_const<T>::type type;
};

// reclector
#define REFLECTABLE(...)                                           \
  static const int fields_n = BOOST_PP_VARIADIC_SIZE(__VA_ARGS__); \
  friend struct reflector;                                         \
  template <int N, class Self>                                     \
  struct field_data                                                \
  {                                                                \
  };                                                               \
  BOOST_PP_SEQ_FOR_EACH_I(REFLECT_EACH, data, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REFLECT_EACH(r, data, i, x)                         \
  PAIR(x);                                                  \
  template <class Self>                                     \
  struct field_data<i, Self>                                \
  {                                                         \
    Self &self;                                             \
    field_data(Self &self) : self(self)                     \
    {                                                       \
    }                                                       \
                                                            \
    typename make_const<Self, TYPEOF(x)>::type &get()       \
    {                                                       \
      return self.STRIP(x);                                 \
    }                                                       \
    typename boost::add_const<TYPEOF(x)>::type &get() const \
    {                                                       \
      return self.STRIP(x);                                 \
    }                                                       \
    const char *name() const                                \
    {                                                       \
      return BOOST_PP_STRINGIZE(STRIP(x));                  \
    }                                                       \
  };

struct reflector
{
  // Get field_data at index N
  template <int N, class T>
  static typename T::template field_data<N, T> get_field_data(T &x)
  {
    return typename T::template field_data<N, T>(x);
  }

  // Get the number of fields
  template <class T>
  struct fields
  {
    static const int n = T::fields_n;
  };
};

struct field_visitor
{
  template <class C, class Visitor, class T>
  void operator()(C &c, Visitor v, T, FILE *os)
  {
    v(reflector::get_field_data<T::value>(c));
  }
};


// fusion struct
namespace BF = boost::fusion;

static auto inline pretty(std::string_view sv) { return std::quoted(sv); }

template <typename T,
          typename Enable = std::enable_if_t<
              not std::is_constructible_v<std::string_view, T const &>>>
static inline T const &pretty(T const &v)
{
  return v;
}

template <typename T,
          typename Enable = std::enable_if_t<
              // BF::traits::is_sequence<T>::type::value>
              std::is_same_v<BF::struct_tag, typename BF::traits::tag_of<T>::type>>>
std::ostream &operator<<(std::ostream &os, T const &v)
{
  bool first = true;
  auto visitor = [&]<size_t I>()
  {
    os << (std::exchange(first, false) ? "\n" : ",\n")
       << BF::extension::struct_member_name<T, I>::call()
       << " : " << pretty(BF::at_c<I>(v));
  };

  // visit members
  [&]<size_t... II>(std::index_sequence<II...>)
  {
    return (visitor.template operator()<II>(), ...);
  }
  (std::make_index_sequence<BF::result_of::size<T>::type::value>{});
  return os;
}

template <typename T,
          typename Enable = std::enable_if_t<
              // BF::traits::is_sequence<T>::type::value>
              std::is_same_v<BF::struct_tag, typename BF::traits::tag_of<T>::type>>>
FILE* fprintf(FILE* f, T const &v)
{
  bool first = true;
  auto visitor = [&]<size_t I>()
  { 
    std::cout << BF::extension::struct_member_name<T, I>::call() << " : " << pretty(BF::at_c<I>(v)) << "\n";
    std::string name_in = BF::extension::struct_member_name<T, I>::call();

    std::stringstream ss;
    ss << pretty(BF::at_c<I>(v));
    std::string value_out;
    ss >> std::quoted(value_out);
    
    fprintf(f, "%s : %s\n", name_in.c_str(), value_out.c_str());
  };

  // visit members
  [&]<size_t... II>(std::index_sequence<II...>)
  {
    return (visitor.template operator()<II>(), ...);
  }
  (std::make_index_sequence<BF::result_of::size<T>::type::value>{});
  return f;
}
